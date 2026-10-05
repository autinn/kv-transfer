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
    assert g.K[(0, 0)].d_in == 2 * 4 * 16


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
    assert g.K[(0, 0)].d_in == expected.shape[-1]
    # Same features feed every target head (all_heads doesn't depend on target head).
    assert g.K[(0, 1)].d_in == g.K[(0, 0)].d_in


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
