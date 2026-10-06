"""all_heads feature mode: target head h reads every source KV head, not just head h.

The paper states the mapper's input concatenates "all Key heads from source layer l_i"
(k*kv_heads*head_dim features), enabling cross-head information flow. This repo's
default reads only the same-numbered source head (k*head_dim). These tests cover the
new `all_heads=True` path in both stream.TopKGrams (fitting) and mapper.KVMapper
(inference), and that the default path is unchanged.
"""

import torch

from kvtransfer.mapper import KVMapper
from kvtransfer.stream import TopKGrams


def test_same_head_feature_width_unchanged():
    """Default (all_heads=False): d_in = k * head_dim, independent of kv_heads."""
    g = TopKGrams({0: [0, 1]}, kv_heads=4, head_dim=16, kinds=("K",))
    assert g.K[(0, 0)].d_in == 2 * 16


def test_all_heads_feature_width():
    """all_heads=True: d_in = k * kv_heads * head_dim."""
    g = TopKGrams({0: [0, 1]}, kv_heads=4, head_dim=16, kinds=("K",), all_heads=True)
    assert g.K[0].d_in == 2 * 4 * 16
    # One shared accumulator per target layer, every head's targets side by side.
    assert g.K[0].d_out == 4 * 16


def test_all_heads_update_matches_manual_concat():
    """The accumulated features are the concatenation across BOTH selected layers AND
    all kv_heads, head order first within a layer, in head-index order."""
    H, D, S, B = 3, 8, 5, 1
    g = TopKGrams({0: [0, 1]}, kv_heads=H, head_dim=D, kinds=("K",), all_heads=True)
    torch.manual_seed(0)
    src_k = [torch.randn(1, B, H, S, D) for _ in range(2)]   # (L,B,H,S,D)
    src_v = [torch.randn(1, B, H, S, D) for _ in range(2)]
    tgt_k = [torch.randn(1, B, H, S, D)]
    tgt_v = [torch.randn(1, B, H, S, D)]
    g.update(torch.cat(src_k, 0), torch.cat(src_v, 0), torch.cat(tgt_k, 0), torch.cat(tgt_v, 0))

    # Manually build the expected feature width for target head 0: all H source heads
    # of source layer 0, then all H source heads of source layer 1.
    expected_cols = []
    for ls in (0, 1):
        sk_l = src_k[ls][0, 0]  # (H, S, D)
        for h2 in range(H):
            expected_cols.append(sk_l[h2])
    expected = torch.cat(expected_cols, dim=-1)  # (S, k*H*D)
    assert g.K[0].d_in == expected.shape[-1]
    # Same features feed every target head, so they share one accumulator: X^T X must
    # equal the manual concatenation's Gram exactly.
    x = expected.to(torch.float64)
    torch.testing.assert_close(g.K[0].xtx, x.T @ x)
    # Targets are every target head side by side, head-major.
    y = torch.cat(list(tgt_k[0][0, 0]), dim=-1).to(torch.float64)   # (S, H*D)
    torch.testing.assert_close(g.K[0].xty, x.T @ y)


def test_mapper_all_heads_roundtrip_and_check_compat():
    D, H = 16, 3
    sel = {0: [0, 1]}
    m = KVMapper(sel, D, H, 1e6, n_src_layers=2, src_layers_v=sel, all_heads=True)
    d_in = 2 * H * D
    for h in range(H):
        m.set("K", 0, h, torch.randn(d_in, D) * 0.1, torch.zeros(D), 0.5)
        m.set("V", 0, h, torch.randn(d_in, D) * 0.1, torch.zeros(D), 0.5)

    sk = [torch.randn(1, H, 4, D) for _ in range(2)]
    sv = [torch.randn(1, H, 4, D) for _ in range(2)]
    k, v = m.map_content(sk, sv)
    assert k[0].shape == (1, H, 4, D) and v[0].shape == (1, H, 4, D)

    m.check_compat(2, 1, H, D)   # fine

    r = KVMapper.from_state_dict(m.state_dict())
    assert r.all_heads is True
    for a, b in zip(m.map_content(sk, sv), r.map_content(sk, sv)):
        for u, w in zip(a, b):
            assert torch.allclose(u, w)


def test_check_compat_rejects_wrong_feature_width():
    """A mapper fitted same-head-only must be refused if treated as all_heads, and
    vice versa -- the W shapes don't match what map_content would build."""
    D, H = 16, 3
    sel = {0: [0, 1]}
    # Fitted same-head (d_in = 2*D) but flagged all_heads=True: width mismatch.
    m = KVMapper(sel, D, H, 1e6, n_src_layers=2, src_layers_v=sel, all_heads=True)
    for h in range(H):
        m.set("K", 0, h, torch.randn(2 * D, D), torch.zeros(D), 0.5)   # wrong width
        m.set("V", 0, h, torch.randn(2 * D, D), torch.zeros(D), 0.5)
    import pytest
    with pytest.raises(ValueError, match="input features"):
        m.check_compat(2, 1, H, D)


def test_all_heads_default_is_false_and_backward_compatible():
    """A state_dict saved before all_heads existed loads as all_heads=False."""
    D, H = 16, 2
    sel = {0: [0]}
    m = KVMapper(sel, D, H, 1e6, n_src_layers=1, src_layers_v=sel)
    for h in range(H):
        m.set("K", 0, h, torch.eye(D), torch.zeros(D), 1.0)
        m.set("V", 0, h, torch.eye(D), torch.zeros(D), 1.0)
    sd = m.state_dict()
    del sd["meta"]["all_heads"]       # simulate a pre-existing saved mapper
    r = KVMapper.from_state_dict(sd)
    assert r.all_heads is False


def _per_head_reference(src_layers, H, D, batches, kind, lam):
    """The pre-sharing implementation: one Gram per (target layer, head), each fed the
    same all-heads features but only that head's targets. The shared accumulator must
    reproduce its solutions exactly."""
    from kvtransfer.gram import GramAccumulator
    from kvtransfer.ridge import solve
    from kvtransfer.stream import _per_head

    accs = {(lt, h): GramAccumulator(len(srcs) * H * D, D)
            for lt, srcs in src_layers.items() for h in range(H)}
    for sk, sv, tk, tv in batches:
        s, t = _per_head(sk if kind == "K" else sv), _per_head(tk if kind == "K" else tv)
        for lt, srcs in src_layers.items():
            x = torch.cat([s[ls, h2] for ls in srcs for h2 in range(H)], dim=-1)
            for h in range(H):
                accs[(lt, h)].update(x, t[lt, h])
    return {key: solve(acc, lam=lam) for key, acc in accs.items()}


def test_all_heads_shared_gram_matches_per_head_grams():
    """Sharing X^T X across target heads is an exact optimisation, not an approximation:
    W, b and R^2 match separate per-head Grams for every head, kind and lambda, across
    several streamed batches and unequal source/target depths."""
    torch.manual_seed(0)
    Ls, Lt, H, D, B, S = 4, 3, 3, 4, 2, 40     # n = B*S*batches = 240 > d_in = 2*H*D = 24
    src_layers = {0: [0, 2], 1: [1, 3], 2: [3, 0]}
    # Correlated targets so the fit is non-trivial (R^2 well above 0).
    mix_k, mix_v = torch.randn(Ls * H * D, Lt * H * D), torch.randn(Ls * H * D, Lt * H * D)
    batches = []
    for _ in range(3):
        sk, sv = torch.randn(Ls, B, H, S, D), torch.randn(Ls, B, H, S, D)
        def tgt(src, mix):
            flat = src.permute(1, 3, 0, 2, 4).reshape(B, S, Ls * H * D) @ mix
            flat = flat + 0.5 * torch.randn_like(flat)
            return flat.reshape(B, S, Lt, H, D).permute(2, 0, 3, 1, 4).contiguous()
        batches.append((sk, sv, tgt(sk, mix_k), tgt(sv, mix_v)))

    g = TopKGrams(src_layers, kv_heads=H, head_dim=D, all_heads=True)
    for b in batches:
        g.update(*b)

    for kind in ("K", "V"):
        for lam in (0.001, 0.01, 0.1):
            ref = _per_head_reference(src_layers, H, D, batches, kind, lam)
            for lt in src_layers:
                shared = g.solve_heads(kind, lt, lam)
                assert len(shared) == H
                for h in range(H):
                    r, s = ref[(lt, h)], shared[h]
                    torch.testing.assert_close(s.W, r.W, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(s.b, r.b, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(s.r2_per_dim, r.r2_per_dim, rtol=1e-5, atol=1e-6)
                    assert s.n == r.n and s.lam == r.lam
                    assert r.r2 > 0.3     # the test data actually has signal to fit


def test_all_heads_shared_gram_footprint_is_one_per_layer():
    """bytes_per_target_layer matches the real allocation: one shared accumulator per kind,
    about kv_heads times smaller than per-head Grams of the same width."""
    k, H, D = 2, 4, 16
    g = TopKGrams({0: list(range(k))}, kv_heads=H, head_dim=D, all_heads=True)
    actual = sum(a.xtx.numel() + a.xty.numel() + a.sx.numel() + a.sy.numel() + a.syy.numel()
                 for a in (g.K[0], g.V[0])) * 8
    assert TopKGrams.bytes_per_target_layer(k, H, D, n_kinds=2, all_heads=True) == actual
    # At the real size (k=8, 8 heads x 128), X^T X dominates and is stored once instead
    # of 8 times: close to an 8x saving.
    k, H, D = 8, 8, 128
    d_in = k * H * D
    per_head = 8 * (d_in * d_in + d_in * D + d_in + 2 * D) * H * 2
    shared = TopKGrams.bytes_per_target_layer(k, H, D, n_kinds=2, all_heads=True)
    assert per_head / shared > 7


def test_topk_grams_state_dict_roundtrip_both_modes():
    torch.manual_seed(1)
    H, D = 2, 4
    batch = (torch.randn(2, 1, H, 30, D), torch.randn(2, 1, H, 30, D),
             torch.randn(1, 1, H, 30, D), torch.randn(1, 1, H, 30, D))
    for all_heads in (False, True):
        g = TopKGrams({0: [0, 1]}, kv_heads=H, head_dim=D, all_heads=all_heads)
        g.update(*batch)
        g2 = TopKGrams.from_state_dict(g.state_dict())
        assert g2.all_heads == all_heads and set(g2.K) == set(g.K)
        for a, b in zip(g.solve_heads("V", 0, 0.01), g2.solve_heads("V", 0, 0.01)):
            torch.testing.assert_close(a.W, b.W)
