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
    ):
        self.src_layers = {int(k): list(v) for k, v in src_layers.items()}
        self.head_dim = head_dim
        self.kv_heads = kv_heads
        self.rope_theta_target = rope_theta_target
        self.K: dict[tuple[int, int], HeadMap] = {}
        self.V: dict[tuple[int, int], HeadMap] = {}

    @property
    def n_tgt_layers(self) -> int:
        return len(self.src_layers)

    def set(self, kind: str, lt: int, h: int, W, b, r2: float) -> None:
        store = self.K if kind == "K" else self.V
        store[(lt, h)] = HeadMap(W.to(torch.float32), b.to(torch.float32), float(r2))

    @torch.no_grad()
    def map_content(
        self, src_k: list[torch.Tensor], src_v: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Map content-space source KV to content-space target KV.

        src_k[l], src_v[l]: (B, H, S, D) pre-RoPE. Returns lists over target layers.
        """
        out_k, out_v = [], []
        for lt in range(self.n_tgt_layers):
            srcs = self.src_layers[lt]
            B, H, S, D = src_k[srcs[0]].shape
            ks = torch.empty(B, H, S, D, dtype=torch.float32)
            vs = torch.empty(B, H, S, D, dtype=torch.float32)
            for h in range(H):
                # (B, S, k*D): concatenate the chosen source layers for this head.
                xk = torch.cat([src_k[ls][:, h].float() for ls in srcs], dim=-1)
                xv = torch.cat([src_v[ls][:, h].float() for ls in srcs], dim=-1)
                mk, mv = self.K[(lt, h)], self.V[(lt, h)]
                ks[:, h] = xk @ mk.W + mk.b
                vs[:, h] = xv @ mv.W + mv.b
            out_k.append(ks)
            out_v.append(vs)
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
            }
        }
        for name, store in (("K", self.K), ("V", self.V)):
            for (lt, h), m in store.items():
                sd[f"{name}.{lt}.{h}"] = {"W": m.W, "b": m.b, "r2": m.r2}
        return sd

    @classmethod
    def from_state_dict(cls, sd: dict) -> "KVMapper":
        m = sd["meta"]
        o = cls(m["src_layers"], m["head_dim"], m["kv_heads"], m["rope_theta_target"])
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
    """

    def __init__(self, n_tgt_layers, head_dim, kv_heads, rope_theta_target):
        super().__init__(
            {l: [l] for l in range(n_tgt_layers)}, head_dim, kv_heads, rope_theta_target
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

    def __init__(self, means_k, means_v, head_dim, kv_heads, rope_theta_target):
        n = len(means_k)
        super().__init__(
            {l: [l] for l in range(n)}, head_dim, kv_heads, rope_theta_target
        )
        zeroW = torch.zeros(head_dim, head_dim)
        for lt in range(n):
            for h in range(kv_heads):
                self.set("K", lt, h, zeroW, means_k[lt][h], float("nan"))
                self.set("V", lt, h, zeroW, means_v[lt][h], float("nan"))
