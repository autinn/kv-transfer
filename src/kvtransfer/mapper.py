"""The fitted cross-model KV mapper.

Holds per-(target layer, head) ridge solutions for keys and values, plus the source
layers feeding each target layer, and applies them to a captured source cache:

    K_hat_t = (K_s R_s^-1(t) W_K + b_K) R_t(t)
    V_hat_t =  V_s W_V + b_V

Both models here share head_dim and rope_theta, so R_s == R_t and the two rotations
cancel at the calibration positions. The strip/re-apply is kept anyway because it is
what makes the fitted W position-independent -- a mapper fitted at positions 0..1023
should still work at position 4096, and only the content-space formulation guarantees
that. It also keeps the code correct for pairs whose RoPE configs differ.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .graft import build_cache
from .select import identity_alignment


@dataclass
class HeadMap:
    W: torch.Tensor    # (k*D, D)
    b: torch.Tensor    # (D,)
    r2: float


class KVMapper:
    """Per-(layer, head) linear maps plus the source-layer selection they were fit with."""

    def __init__(
        self,
        src_layers: dict[int, list[int]],
        head_dim: int,
        kv_heads: int,
        rope_theta_target: float,
        n_src_layers: int | None = None,
        src_layers_v: dict[int, list[int]] | None = None,
        all_heads: bool = False,
    ):
        self.src_layers = {int(k): list(v) for k, v in src_layers.items()}
        # Values may read different source layers than keys: top-k selection ranks K
        # and V separately. Defaults to the key selection.
        self.src_layers_v = (
            self.src_layers if src_layers_v is None
            else {int(k): list(v) for k, v in src_layers_v.items()}
        )
        if sorted(self.src_layers_v) != sorted(self.src_layers):
            raise ValueError("key and value selections must cover the same target layers")
        self.head_dim = head_dim
        self.kv_heads = kv_heads
        self.rope_theta_target = rope_theta_target
        # Recorded so a mapper fitted for one pair is refused on another; None for
        # mappers saved before this field existed.
        self.n_src_layers = n_src_layers
        # When True, every target head's feature vector concatenates ALL source KV
        # heads of each selected layer (k*kv_heads*head_dim inputs), matching the
        # paper's stated construction, instead of only the same-numbered source head
        # (k*head_dim). See stream.TopKGrams for the matching Stage B feature code.
        self.all_heads = all_heads
        self.K: dict[tuple[int, int], HeadMap] = {}
        self.V: dict[tuple[int, int], HeadMap] = {}
        self.device = torch.device("cpu")
        # Precision map_content computes in. float32 for every fit and eval; bfloat16
        # only for timing the transfer the way a serving system would run it.
        self.dtype = torch.float32
        # all_heads: per target layer, the 8 heads' W stacked side by side (built on
        # first use, dropped whenever the maps change).
        self._stacked: dict | None = None

    @property
    def n_tgt_layers(self) -> int:
        return len(self.src_layers)

    def set(self, kind: str, lt: int, h: int, W, b, r2: float) -> None:
        store = self.K if kind == "K" else self.V
        store[(lt, h)] = HeadMap(
            W.to(self.device, self.dtype), b.to(self.device, self.dtype), float(r2)
        )
        self._stacked = None

    def to(self, device, dtype: torch.dtype | None = None) -> "KVMapper":
        """Move the maps to `device` (and optionally `dtype`); `map_content` then runs
        there, in that precision."""
        self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        for store in (self.K, self.V):
            for key, m in store.items():
                store[key] = HeadMap(m.W.to(self.device, self.dtype), m.b.to(self.device, self.dtype), m.r2)
        self._stacked = None
        return self

    def _stack(self) -> dict:
        """(kind, target layer) -> (W, b) with every head's map side by side:
        W (k*H*D, H*D), b (H*D). In all_heads mode every head of a layer reads the same
        features, so one matmul against this replaces H per-head matmuls exactly."""
        if self._stacked is None:
            self._stacked = {
                (kind, lt): (torch.cat([store[(lt, h)].W for h in range(self.kv_heads)], dim=1),
                             torch.cat([store[(lt, h)].b for h in range(self.kv_heads)]))
                for kind, store in (("K", self.K), ("V", self.V))
                for lt in range(self.n_tgt_layers)
            }
        return self._stacked

    def check_compat(self, n_src_layers: int, n_tgt_layers: int, kv_heads: int, head_dim: int) -> None:
        """Refuse to apply a mapper to a model pair it was not fitted for."""
        problems = []
        if self.n_tgt_layers != n_tgt_layers:
            problems.append(f"maps {self.n_tgt_layers} target layers, target has {n_tgt_layers}")
        if self.n_src_layers is not None and self.n_src_layers != n_src_layers:
            problems.append(f"fitted on {self.n_src_layers} source layers, source has {n_src_layers}")
        top = max(max(v) for sel in (self.src_layers, self.src_layers_v) for v in sel.values())
        if top >= n_src_layers:
            problems.append(f"reads source layer {top}, source has {n_src_layers}")
        if (self.kv_heads, self.head_dim) != (kv_heads, head_dim):
            problems.append(f"shape {self.kv_heads}x{self.head_dim} vs model {kv_heads}x{head_dim}")
        missing = [(lt, h) for lt in range(self.n_tgt_layers) for h in range(self.kv_heads)
                   if (lt, h) not in self.K or (lt, h) not in self.V]
        if missing:
            problems.append(f"{len(missing)} (layer, head) maps missing, e.g. {missing[0]}")
        else:
            # Catch a W fitted for one feature width (same-head vs all-heads) being
            # applied under the other -- map_content would otherwise fail deep inside
            # a matmul with a confusing shape-mismatch error.
            mult = self.kv_heads if self.all_heads else 1
            for name, sel, store in (("K", self.src_layers, self.K), ("V", self.src_layers_v, self.V)):
                for lt, srcs in sel.items():
                    expected = len(srcs) * head_dim * mult
                    got = store[(lt, 0)].W.shape[0]
                    if got != expected:
                        problems.append(
                            f"{name} map at target layer {lt} has {got} input features, "
                            f"expected {expected} (all_heads={self.all_heads})"
                        )
                        break
        if problems:
            raise ValueError("mapper does not fit this model pair: " + "; ".join(problems))

    @torch.no_grad()
    def map_content(
        self, src_k: list[torch.Tensor], src_v: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Map content-space source KV to content-space target KV.

        src_k[l], src_v[l]: (B, H, S, D) pre-RoPE, one entry per SOURCE layer. Returns
        lists over TARGET layers, on the mapper's device. The two counts differ when the
        models have different depths.
        """
        dev = self.device
        needed = sorted({ls for sel in (self.src_layers, self.src_layers_v)
                         for srcs in sel.values() for ls in srcs})
        if needed[-1] >= len(src_k) or len(src_k) != len(src_v):
            raise ValueError(
                f"mapper reads source layer {needed[-1]} but got {len(src_k)} key / "
                f"{len(src_v)} value layers"
            )
        # Each source layer may feed several target layers; move it to the device once.
        sk = {ls: src_k[ls].to(dev, self.dtype) for ls in needed}
        sv = {ls: src_v[ls].to(dev, self.dtype) for ls in needed}
        if self.all_heads:
            return self._map_all_heads(sk, sv)
        out_k, out_v = [], []
        for lt in range(self.n_tgt_layers):
            srcs, srcs_v = self.src_layers[lt], self.src_layers_v[lt]
            B, H, S, D = sk[srcs[0]].shape
            ks = torch.empty(B, H, S, D, dtype=self.dtype, device=dev)
            vs = torch.empty(B, H, S, D, dtype=self.dtype, device=dev)
            for h in range(H):
                # (B, S, k*D): head h of each selected source layer.
                xk = torch.cat([sk[ls][:, h] for ls in srcs], dim=-1)
                xv = torch.cat([sv[ls][:, h] for ls in srcs_v], dim=-1)
                mk, mv = self.K[(lt, h)], self.V[(lt, h)]
                ks[:, h] = xk @ mk.W + mk.b
                vs[:, h] = xv @ mv.W + mv.b
            out_k.append(ks)
            out_v.append(vs)
        return out_k, out_v

    def _map_all_heads(self, sk: dict, sv: dict):
        """all_heads mapping, one matmul per (target layer, kind).

        The feature vector is every source head of each selected layer, layer-major then
        head then dim -- the order stream.TopKGrams.update fits in. Each source layer is
        flattened to (B, S, H*D) once, however many target layers read it.
        """
        def flat(x):  # (B, H, S, D) -> (B, S, H*D), head-major within a position
            B, H, S, D = x.shape
            return x.transpose(1, 2).reshape(B, S, H * D)

        fk, fv = {ls: flat(x) for ls, x in sk.items()}, {ls: flat(x) for ls, x in sv.items()}
        stacked = self._stack()
        H, D = self.kv_heads, self.head_dim
        out_k, out_v = [], []
        for lt in range(self.n_tgt_layers):
            for kind, feats, sel, out in (("K", fk, self.src_layers, out_k),
                                          ("V", fv, self.src_layers_v, out_v)):
                srcs = sel[lt]
                x = feats[srcs[0]] if len(srcs) == 1 else torch.cat([feats[ls] for ls in srcs], dim=-1)
                W, b = stacked[(kind, lt)]
                y = torch.addmm(b, x.reshape(-1, x.shape[-1]), W)        # (B*S, H*D)
                B, S = x.shape[0], x.shape[1]
                out.append(y.view(B, S, H, D).transpose(1, 2).contiguous())
        return out_k, out_v

    @torch.no_grad()
    def to_cache(self, src_k, src_v, offset: int = 0, dtype=None, device=None, config=None):
        """Map source KV and build a target-model cache (applies the target's RoPE)."""
        k, v = self.map_content(src_k, src_v)
        return build_cache(
            k, v, self.head_dim, self.rope_theta_target,
            offset=offset, dtype=dtype, device=device, config=config,
        )

    def mean_r2(self, kind: str = "K") -> float:
        store = self.K if kind == "K" else self.V
        return sum(m.r2 for m in store.values()) / max(len(store), 1)

    def state_dict(self) -> dict:
        sd = {
            "meta": {
                "src_layers": self.src_layers, "head_dim": self.head_dim,
                "kv_heads": self.kv_heads, "rope_theta_target": self.rope_theta_target,
                "n_src_layers": self.n_src_layers,
                "src_layers_v": self.src_layers_v,
                "all_heads": self.all_heads,
            }
        }
        for name, store in (("K", self.K), ("V", self.V)):
            for (lt, h), m in store.items():
                sd[f"{name}.{lt}.{h}"] = {"W": m.W.cpu(), "b": m.b.cpu(), "r2": m.r2}
        return sd

    @classmethod
    def from_state_dict(cls, sd: dict) -> "KVMapper":
        m = sd["meta"]
        if "src_layers_v" not in m and any(len(v) > 1 for v in m["src_layers"].values()):
            import warnings
            warnings.warn(
                "mapper saved before value selections were recorded: its V maps may have "
                "been fitted on different source layers than the ones it will read. Refit it.",
                stacklevel=2,
            )
        o = KVMapper(
            m["src_layers"], m["head_dim"], m["kv_heads"], m["rope_theta_target"],
            n_src_layers=m.get("n_src_layers"), src_layers_v=m.get("src_layers_v"),
            all_heads=m.get("all_heads", False),
        )
        for key, val in sd.items():
            if key == "meta":
                continue
            name, lt, h = key.split(".")
            o.set(name, int(lt), int(h), val["W"], val["b"], val["r2"])
        return o


class ZeroCopyMapper(KVMapper):
    """Control: paste the source cache in unchanged (identity map, no fitting).

    The point of comparison the retention number needs. If a fitted mapper does not
    clearly beat this, the linear map has not learned anything about the cross-model
    relationship -- the models were simply already similar enough.

    Target layer lt reads source layer round(lt * (Ls-1)/(Lt-1)): layer l -> l when the
    depths match, proportional when they do not (14B's 40 layers spread over 32B's 64).
    """

    def __init__(self, n_tgt_layers, head_dim, kv_heads, rope_theta_target, n_src_layers=None):
        n_src_layers = n_tgt_layers if n_src_layers is None else n_src_layers
        super().__init__(
            identity_alignment(n_src_layers, n_tgt_layers), head_dim, kv_heads,
            rope_theta_target, n_src_layers=n_src_layers,
        )
        eye = torch.eye(head_dim)
        zero = torch.zeros(head_dim)
        for lt in range(n_tgt_layers):
            for h in range(kv_heads):
                self.set("K", lt, h, eye, zero, float("nan"))
                self.set("V", lt, h, eye, zero, float("nan"))


class MeanMapper(KVMapper):
    """Floor: predict the target's per-channel mean, ignoring the source entirely.

    This is what R^2 = 0 looks like end-to-end. Any honest retention number must be
    compared against it, because a cache of per-channel means is not random noise and
    can still score non-trivially on multiple-choice benchmarks.
    """

    def __init__(self, means_k, means_v, head_dim, kv_heads, rope_theta_target, n_src_layers=None):
        n = len(means_k)
        # W is zero, so which source layer is "read" is irrelevant; layer 0 always exists.
        super().__init__(
            {l: [0] for l in range(n)}, head_dim, kv_heads, rope_theta_target,
            n_src_layers=n_src_layers,
        )
        zeroW = torch.zeros(head_dim, head_dim)
        for lt in range(n):
            for h in range(kv_heads):
                self.set("K", lt, h, zeroW, means_k[lt][h], float("nan"))
                self.set("V", lt, h, zeroW, means_v[lt][h], float("nan"))
