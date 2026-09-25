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

    `src_layers[lt]` lists the source layers feeding target layer lt, so the feature
    vector for target (lt, h) is the concatenation of those source layers' head-h
    vectors -- the paper's X^l = [K_s^(l1) || ... || K_s^(lk)].
    """

    def __init__(
        self,
        src_layers: dict[int, list[int]],
        kv_heads: int,
        head_dim: int,
        device="cpu",
        kinds: tuple[str, ...] = ("K", "V"),
    ):
        self.src_layers = {int(k): list(v) for k, v in src_layers.items()}
        self.kv_heads, self.head_dim = kv_heads, head_dim
        self.kinds = tuple(kinds)
        self.K, self.V = {}, {}
        for lt, srcs in self.src_layers.items():
            d_in = len(srcs) * head_dim
            for h in range(kv_heads):
                if "K" in self.kinds:
                    self.K[(lt, h)] = GramAccumulator(d_in, head_dim, device)
                if "V" in self.kinds:
                    self.V[(lt, h)] = GramAccumulator(d_in, head_dim, device)

    @staticmethod
    def bytes_per_target_layer(k: int, kv_heads: int, head_dim: int, n_kinds: int = 2) -> int:
        """fp64 accumulator footprint of one target layer, for grouping under a budget."""
        d_in = k * head_dim
        per = d_in * d_in + d_in * head_dim + d_in + 2 * head_dim
        return 8 * per * kv_heads * n_kinds

    @torch.no_grad()
    def update(self, src_k, src_v, tgt_k, tgt_v) -> None:
        """Args (L, B, H, S, D); source and target layer counts may differ."""
        sk, sv = _per_head(src_k), _per_head(src_v)
        tk, tv = _per_head(tgt_k), _per_head(tgt_v)
        for lt, srcs in self.src_layers.items():
            for h in range(self.kv_heads):
                # concat chosen source layers along the feature dim -> (N, k*D)
                if "K" in self.kinds:
                    xk = torch.cat([sk[ls, h] for ls in srcs], dim=-1)
                    self.K[(lt, h)].update(xk, tk[lt, h])
                if "V" in self.kinds:
                    xv = torch.cat([sv[ls, h] for ls in srcs], dim=-1)
                    self.V[(lt, h)].update(xv, tv[lt, h])

    def state_dict(self) -> dict:
        sd = {"meta": {"src_layers": self.src_layers, "kv_heads": self.kv_heads,
                       "head_dim": self.head_dim, "kinds": list(self.kinds)}}
        for name, store in (("K", self.K), ("V", self.V)):
            for (lt, h), acc in store.items():
                sd[f"{name}.{lt}.{h}"] = acc.state_dict()
        return sd

    @classmethod
    def from_state_dict(cls, sd: dict) -> "TopKGrams":
        m = sd["meta"]
        o = cls(m["src_layers"], m["kv_heads"], m["head_dim"], kinds=tuple(m.get("kinds", ("K", "V"))))
        for key, val in sd.items():
            if key == "meta":
                continue
            name, lt, h = key.split(".")
            (o.K if name == "K" else o.V)[(int(lt), int(h))] = GramAccumulator.from_state_dict(val)
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
