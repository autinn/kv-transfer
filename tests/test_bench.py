"""Speedup harness (bench.py): the timed transfer is the real graft path, at full shape.

A timing run proves nothing if the thing timed is not what the eval runs, so these
check that the random-weight mapper has the fitted shape, that `transfer` produces the
cache the target model expects, and that the bf16 path computes the same map.
"""

import torch

from kvtransfer.bench import (
    body_params, mapper_flops, prefill, prefill_flops, random_mapper, synthetic_cache,
    time_ms, transfer,
)
from kvtransfer.graft import cache_to_content


def test_random_mapper_has_fitted_shape(tiny_pair):
    src, tgt = tiny_pair
    Ls, Lt = src.config.num_hidden_layers, tgt.config.num_hidden_layers
    H, D = tgt.config.num_key_value_heads, tgt.config.head_dim
    m = random_mapper(Ls, Lt, H, D, 1e6, k=2, all_heads=True)
    m.check_compat(Ls, Lt, H, D)  # raises if any map is missing or the wrong width
    assert all(len(v) == 2 for v in m.src_layers.values())
    assert m.K[(0, 0)].W.shape == (2 * H * D, D)


def test_transfer_builds_target_shaped_cache(tiny_pair):
    src, tgt = tiny_pair
    Ls, Lt = src.config.num_hidden_layers, tgt.config.num_hidden_layers
    H, D = tgt.config.num_key_value_heads, tgt.config.head_dim
    ids = torch.randint(0, 97, (1, 12))
    m = random_mapper(Ls, Lt, H, D, 1e6, k=2)
    cache = transfer(m, prefill(src, ids), D, 1e6, torch.float32, tgt.config)
    k, v = cache_to_content(cache, D, 1e6)
    assert len(k) == Lt and k[0].shape == (1, H, 12, D)
    # The target model accepts it and continues decoding from position 12.
    out = tgt(input_ids=torch.randint(0, 97, (1, 1)), past_key_values=cache, use_cache=True)
    assert out.logits.shape == (1, 1, 97)


def test_bf16_mapper_matches_float32():
    m32 = random_mapper(3, 4, 2, 16, 1e6, k=2)
    c = synthetic_cache(3, 2, 16, 10, 1e6, dtype=torch.float32)
    k, v = cache_to_content(c, 16, 1e6)
    a, _ = m32.map_content(k, v)
    m16 = random_mapper(3, 4, 2, 16, 1e6, k=2).to("cpu", torch.bfloat16)
    b, _ = m16.map_content(k, v)
    assert b[0].dtype == torch.bfloat16
    rel = (a[0] - b[0].float()).norm() / a[0].norm()
    assert rel < 2e-2


def test_flop_counts_and_timer(tiny_pair):
    src, _ = tiny_pair
    n = body_params(src)
    assert 0 < n < sum(p.numel() for p in src.parameters())
    # Doubling the length doubles the linear term and quadruples attention.
    f1, f2 = (prefill_flops(n, 2, 4, 16, S) for S in (100, 200))
    assert 2 * f1 < f2 < 4 * f1
    m = random_mapper(2, 3, 2, 16, 1e6, k=1)
    assert mapper_flops(m, 10) == 2 * 10 * (2 * 3 * 2) * (2 * 16) * 16
    r = time_ms(lambda: torch.ones(4) * 2, warmup=1, trials=3)
    assert r["n"] == 3 and r["min"] <= r["median"]
