"""Latency of KV transfer vs re-prefill: the paper's speedup measurement (its §4.7, App. G).

The paper compares the target model re-reading a prompt (its transformer body, no LM
head) against the transfer: the source cache mapped into the target's format. Both run
on synthetic inputs, since latency depends on shapes, not values. That is why
`random_mapper` is enough here: a mapper with random weights of the fitted shape does
exactly the same arithmetic as a fitted one, so no refit is needed to time a pair.

Transfer here means what this repo's graft path does to an existing source cache:
strip the source RoPE (`cache_to_content`), apply the per-(layer, head) maps, re-apply
the target RoPE and build the target `DynamicCache` (`KVMapper.to_cache`).
"""

from __future__ import annotations

import statistics
import time

import torch

from .graft import build_cache, cache_to_content
from .mapper import KVMapper


def random_mapper(n_src: int, n_tgt: int, kv_heads: int, head_dim: int, rope_theta: float,
                  k: int, all_heads: bool = True, device="cpu", seed: int = 0) -> KVMapper:
    """A mapper with the fitted shape and random weights.

    Target layer l reads the k source layers nearest its proportional position, so the
    source layers touched (and the bytes moved) match a real top-k selection's spread.
    """
    g = torch.Generator().manual_seed(seed)
    k = min(k, n_src)
    sel = {}
    for lt in range(n_tgt):
        centre = lt * (n_src - 1) / max(n_tgt - 1, 1)
        sel[lt] = sorted(sorted(range(n_src), key=lambda ls: (abs(ls - centre), ls))[:k])
    m = KVMapper(sel, head_dim, kv_heads, rope_theta, n_src_layers=n_src,
                 src_layers_v=sel, all_heads=all_heads)
    m.to(device)
    d_in = k * head_dim * (kv_heads if all_heads else 1)
    for kind in ("K", "V"):
        for lt in range(n_tgt):
            for h in range(kv_heads):
                W = torch.randn(d_in, head_dim, generator=g) / d_in ** 0.5
                m.set(kind, lt, h, W, torch.zeros(head_dim), 0.0)
    return m


def synthetic_cache(n_layers: int, kv_heads: int, head_dim: int, seq_len: int, rope_theta: float,
                    dtype=torch.bfloat16, device="cpu", seed: int = 0):
    """A random post-RoPE KV cache shaped like a source model's prefill output."""
    g = torch.Generator(device=device).manual_seed(seed)
    shape = (1, kv_heads, seq_len, head_dim)
    keys = [torch.randn(shape, generator=g, device=device, dtype=torch.float32) for _ in range(n_layers)]
    vals = [torch.randn(shape, generator=g, device=device, dtype=torch.float32).to(dtype)
            for _ in range(n_layers)]
    return build_cache(keys, vals, head_dim, rope_theta, dtype=dtype, device=device)


@torch.no_grad()
def transfer(mapper: KVMapper, src_cache, head_dim: int, src_theta: float, dtype, config=None):
    """Source cache -> target cache: strip source RoPE, map, re-apply target RoPE."""
    k, v = cache_to_content(src_cache, head_dim, src_theta)
    return mapper.to_cache(k, v, dtype=dtype, config=config)


@torch.no_grad()
def prefill(model, input_ids):
    """The transformer body over the prompt, producing its KV cache (no LM head)."""
    return model.model(input_ids=input_ids, use_cache=True).past_key_values


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def time_ms(fn, warmup: int, trials: int, device="cpu") -> dict:
    """Wall time of `fn()` in ms over `trials` runs after `warmup` untimed runs.

    Synchronises the GPU around each run so queued kernels are counted. The result of
    each call is dropped before the next, so caches do not pile up in memory.
    """
    for _ in range(warmup):
        out = fn()
        del out
    _sync(device)
    times = []
    for _ in range(trials):
        _sync(device)
        t0 = time.perf_counter()
        out = fn()
        _sync(device)
        times.append(1e3 * (time.perf_counter() - t0))
        del out
    return {"mean": statistics.fmean(times), "median": statistics.median(times),
            "std": statistics.pstdev(times), "min": min(times), "n": trials}


def body_params(model) -> int:
    """Parameters the prefill multiplies by: the transformer body without embeddings."""
    emb = model.get_input_embeddings().weight.numel()
    return sum(p.numel() for p in model.model.parameters()) - emb


def prefill_flops(n_body: int, n_layers: int, n_q_heads: int, head_dim: int, seq_len: int) -> float:
    """~2·N·S for the weight matmuls plus 2·S²·heads·head_dim per layer for causal
    attention (QKᵀ and AV, each over half the S×S grid)."""
    return 2 * n_body * seq_len + n_layers * 2 * seq_len ** 2 * n_q_heads * head_dim


def mapper_flops(mapper: KVMapper, seq_len: int) -> float:
    """2·S·d_in·d_out summed over every (kind, target layer, head) map."""
    return 2 * seq_len * sum(m.W.shape[0] * m.W.shape[1] for store in (mapper.K, mapper.V)
                             for m in store.values())
