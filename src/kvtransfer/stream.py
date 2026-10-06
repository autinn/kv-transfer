"""Streaming accumulation of cross-model KV statistics.

Batches of captured activations (read back from `kvstore` shards) are folded into
normal-equation statistics and dropped, so memory holds only the statistics and one
shard at a time.

Two stages, because accumulating against the full source stack at once does not fit:
a (28*128)^2 fp64 Gram is 103 MB, and one per (target layer, head, K/V) is **46 GB**.

    Stage A (`SingleSourceGrams`): every (src_layer, tgt_layer, head) cell with
        d_in = head_dim. 1.6 GB resident, one pass, and it yields the paper's
        28x28 R^2 linear-structure heatmap directly.

    Stage B (`TopKGrams`): the production fit, using only the top-k source layers
        chosen from Stage A. At k=8 that is 3.8 GB across all target layers, or
        1.6 GB if looped one target layer at a time.

Stage A is not overhead -- it is the paper's central evidence *and* the input to
cross-layer source selection. Stage B then costs one more pass with k fixed.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import torch

from .gram import GramAccumulator
from .ridge import RidgeSolution, solve


@dataclass
class StreamConfig:
    stride: int = 4                       # paper subsamples tokens by 4
    store_dtype: torch.dtype = torch.float32
    device: str = "cpu"
    gram_device: str = "cpu"


def subsample(x: torch.Tensor, stride: int, seq_dim: int = 3) -> torch.Tensor:
    """Take every `stride`-th token, reducing correlation between observations."""
    if stride <= 1:
        return x
    idx = torch.arange(0, x.shape[seq_dim], stride, device=x.device)
    return x.index_select(seq_dim, idx)


def _per_head(x: torch.Tensor) -> torch.Tensor:
    """(L, B, H, S, D) -> (L, H, B*S, D): tokens as observations, per layer and head."""
    L, B, H, S, D = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(L, H, B * S, D)


class SingleSourceGrams:
    """Stage A: one Gram per (src_layer, tgt_layer, head), d_in = head_dim.

    Head-matched: target head h regresses on source head h. Both Qwen3 models have
    8 KV heads, and matched-KV pairs in the paper share head counts, so index
    correspondence is the natural hypothesis -- and Stage A's R^2 map is exactly what
    would expose it if the hypothesis were wrong (a permuted or diffuse heatmap).
    """

    def __init__(self, n_src: int, n_tgt: int, kv_heads: int, head_dim: int, device="cpu"):
        self.n_src, self.n_tgt = n_src, n_tgt
        self.kv_heads, self.head_dim = kv_heads, head_dim
        mk = lambda: [
            [[GramAccumulator(head_dim, head_dim, device) for _ in range(kv_heads)]
             for _ in range(n_tgt)]
            for _ in range(n_src)
        ]
        self.K = mk()
        self.V = mk()

    @torch.no_grad()
    def update(self, src_k, src_v, tgt_k, tgt_v) -> None:
        """All args (L, B, H, S, D) in content space (pre-RoPE)."""
        sk, sv = _per_head(src_k), _per_head(src_v)   # (Ls, H, N, D)
        tk, tv = _per_head(tgt_k), _per_head(tgt_v)   # (Lt, H, N, D)
        for ls in range(self.n_src):
            for lt in range(self.n_tgt):
                for h in range(self.kv_heads):
                    self.K[ls][lt][h].update(sk[ls, h], tk[lt, h])
                    self.V[ls][lt][h].update(sv[ls, h], tv[lt, h])

    def state_dict(self) -> dict:
        sd = {"meta": {"n_src": self.n_src, "n_tgt": self.n_tgt,
                       "kv_heads": self.kv_heads, "head_dim": self.head_dim}}
        for name, store in (("K", self.K), ("V", self.V)):
            for ls in range(self.n_src):
                for lt in range(self.n_tgt):
                    for h in range(self.kv_heads):
                        sd[f"{name}.{ls}.{lt}.{h}"] = store[ls][lt][h].state_dict()
        return sd

    @classmethod
    def from_state_dict(cls, sd: dict) -> "SingleSourceGrams":
        m = sd["meta"]
        o = cls(m["n_src"], m["n_tgt"], m["kv_heads"], m["head_dim"])
        for key, val in sd.items():
            if key == "meta":
                continue
            name, ls, lt, h = key.split(".")
            store = o.K if name == "K" else o.V
            store[int(ls)][int(lt)][int(h)] = GramAccumulator.from_state_dict(val)
        return o


class TopKGrams:
    """Stage B: production Grams over a chosen top-k source-layer set per target layer.

    `src_layers[lt]` lists the source layers feeding target layer lt. By default the
    feature vector for target (lt, h) concatenates those source layers' head-h vectors
    only -- the paper's X^l = [K_s^(l1) || ... || K_s^(lk)] read literally as one source
    head per target head.

    `all_heads=True` instead concatenates *all* source KV heads of each selected layer
    (d_in = k * kv_heads * head_dim), matching the paper's stated feature construction:
    "the concatenation of all Key heads from source layer l_i," enabling cross-head
    information flow. This is the feature width implied by the paper's quoted mapper
    parameter counts; the default (same-numbered head only) is cheaper and was this
    repo's original reading, now confirmed narrower than the paper's.

    Storage differs by mode. Same-head: one accumulator per (lt, h), keyed `(lt, h)`.
    All-heads: every target head of a layer sees the same features, so X^T X, sum(X)
    and n are identical across heads; one accumulator per lt (keyed `lt`) holds them
    once, with all heads' targets side by side (d_out = kv_heads * head_dim, head-major).
    Ridge solves each output column independently, so this gives the same per-head
    solutions as separate Grams at 1/kv_heads the accumulation cost and memory. Use
    `solve_heads` rather than indexing `K`/`V` directly to stay mode-agnostic.
    """

    def __init__(
        self,
        src_layers: dict[int, list[int]],
        kv_heads: int,
        head_dim: int,
        device="cpu",
        kinds: tuple[str, ...] = ("K", "V"),
        all_heads: bool = False,
    ):
        self.src_layers = {int(k): list(v) for k, v in src_layers.items()}
        self.kv_heads, self.head_dim = kv_heads, head_dim
        self.kinds = tuple(kinds)
        self.all_heads = all_heads
        self.K, self.V = {}, {}
        for lt, srcs in self.src_layers.items():
            d_in = len(srcs) * head_dim * (kv_heads if all_heads else 1)
            if all_heads:
                if "K" in self.kinds:
                    self.K[lt] = GramAccumulator(d_in, kv_heads * head_dim, device)
                if "V" in self.kinds:
                    self.V[lt] = GramAccumulator(d_in, kv_heads * head_dim, device)
                continue
            for h in range(kv_heads):
                if "K" in self.kinds:
                    self.K[(lt, h)] = GramAccumulator(d_in, head_dim, device)
                if "V" in self.kinds:
                    self.V[(lt, h)] = GramAccumulator(d_in, head_dim, device)

    @staticmethod
    def bytes_per_target_layer(
        k: int, kv_heads: int, head_dim: int, n_kinds: int = 2, all_heads: bool = False
    ) -> int:
        """fp64 accumulator footprint of one target layer, for grouping under a budget."""
        if all_heads:
            # One shared accumulator per kind; see the class docstring.
            d_in, d_out = k * kv_heads * head_dim, kv_heads * head_dim
            return 8 * (d_in * d_in + d_in * d_out + d_in + 2 * d_out) * n_kinds
        d_in = k * head_dim
        per = d_in * d_in + d_in * head_dim + d_in + 2 * head_dim
        return 8 * per * kv_heads * n_kinds

    def solve_heads(self, kind: str, lt: int, lam: float) -> list[RidgeSolution]:
        """Per-head ridge solutions for target layer lt, in head order, either mode."""
        store = self.K if kind == "K" else self.V
        if not self.all_heads:
            return [solve(store[(lt, h)], lam=lam) for h in range(self.kv_heads)]
        sol = solve(store[lt], lam=lam)
        D = self.head_dim
        return [
            RidgeSolution(W=sol.W[:, h * D:(h + 1) * D].contiguous(),
                          b=sol.b[h * D:(h + 1) * D].contiguous(), lam=sol.lam,
                          r2_per_dim=sol.r2_per_dim[h * D:(h + 1) * D].contiguous(), n=sol.n)
            for h in range(self.kv_heads)
        ]

    @torch.no_grad()
    def update(self, src_k, src_v, tgt_k, tgt_v) -> None:
        """Args (L, B, H, S, D); source and target layer counts may differ."""
        sk, sv = _per_head(src_k), _per_head(src_v)
        tk, tv = _per_head(tgt_k), _per_head(tgt_v)
        for lt, srcs in self.src_layers.items():
            # All-heads features don't depend on the target head, so build them once
            # per target layer rather than once per (target layer, head).
            if self.all_heads:
                # One shared accumulator per kind, targets of every head side by side.
                if "K" in self.kinds:
                    xk = torch.cat([sk[ls, h2] for ls in srcs for h2 in range(self.kv_heads)], dim=-1)
                    self.K[lt].update(xk, torch.cat(list(tk[lt]), dim=-1))
                if "V" in self.kinds:
                    xv = torch.cat([sv[ls, h2] for ls in srcs for h2 in range(self.kv_heads)], dim=-1)
                    self.V[lt].update(xv, torch.cat(list(tv[lt]), dim=-1))
                continue
            for h in range(self.kv_heads):
                if "K" in self.kinds:
                    xk = torch.cat([sk[ls, h] for ls in srcs], dim=-1)
                    self.K[(lt, h)].update(xk, tk[lt, h])
                if "V" in self.kinds:
                    xv = torch.cat([sv[ls, h] for ls in srcs], dim=-1)
                    self.V[(lt, h)].update(xv, tv[lt, h])

    def state_dict(self) -> dict:
        sd = {"meta": {"src_layers": self.src_layers, "kv_heads": self.kv_heads,
                       "head_dim": self.head_dim, "kinds": list(self.kinds),
                       "all_heads": self.all_heads}}
        for name, store in (("K", self.K), ("V", self.V)):
            for key, acc in store.items():
                # all_heads keys are lt (shared accumulator), same-head keys are (lt, h).
                tail = f"{key[0]}.{key[1]}" if isinstance(key, tuple) else f"{key}"
                sd[f"{name}.{tail}"] = acc.state_dict()
        return sd

    @classmethod
    def from_state_dict(cls, sd: dict) -> "TopKGrams":
        m = sd["meta"]
        o = cls(m["src_layers"], m["kv_heads"], m["head_dim"], kinds=tuple(m.get("kinds", ("K", "V"))),
                all_heads=m.get("all_heads", False))
        for key, val in sd.items():
            if key == "meta":
                continue
            name, *idx = key.split(".")
            k = int(idx[0]) if len(idx) == 1 else (int(idx[0]), int(idx[1]))
            (o.K if name == "K" else o.V)[k] = GramAccumulator.from_state_dict(val)
        return o


@torch.no_grad()
def capture_all(model_loader, batches, cfg: StreamConfig):
    """Load a model, capture pre-RoPE KV for each batch, free the model.

    `model_loader` is a zero-arg callable so the model exists only inside this scope:
    on an 8 GB machine the two models must never be resident simultaneously. Returns a
    list of (keys, values) with shapes (L, B, H, S, D), already stride-subsampled.
    """
    from .capture import KVCapture

    model = model_loader()
    model.eval()
    out = []
    try:
        for ids in batches:
            ids = ids.to(cfg.device)
            with KVCapture(model, store_dtype=cfg.store_dtype) as cap:
                model(input_ids=ids, use_cache=False)
                k, v = cap.stacked()
            out.append((subsample(k, cfg.stride), subsample(v, cfg.stride)))
            cap.clear()
    finally:
        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    return out
