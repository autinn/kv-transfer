"""Source and target with different depths (fix for 14B: 40 layers -> 32B: 64).

Everything that used to assume target layer l reads source layer l: zero-copy, the
aligned "diagonal", source capture in eval, and mapper validation.
"""

import warnings

import pytest
import torch

from kvtransfer.capture import KVCapture
from kvtransfer.eval.ppl import source_prefill_content, transfer_ppl
from kvtransfer.mapper import KVMapper, MeanMapper, ZeroCopyMapper
from kvtransfer.select import aligned_cells, identity_alignment, top_k


def _capture(model, ids):
    with torch.no_grad(), KVCapture(model, store_dtype=torch.float32) as cap:
        model(input_ids=ids, use_cache=False)
        return [cap.keys[l] for l in cap.layers], [cap.values[l] for l in cap.layers]


def test_identity_alignment_is_proportional():
    assert identity_alignment(28, 28) == {l: [l] for l in range(28)}
    a = identity_alignment(40, 64)
    assert a[0] == [0] and a[63] == [39]
    assert all(0 <= v[0] < 40 for v in a.values())
    assert [v[0] for v in a.values()] == sorted(v[0] for v in a.values())


def test_aligned_cells_equals_diagonal_for_equal_depth():
    m = torch.rand(5, 5, 3)
    assert torch.equal(aligned_cells(m), torch.stack([m[i, i] for i in range(5)]))


def test_aligned_cells_unequal_depth():
    m = torch.arange(4 * 7, dtype=torch.float32).reshape(4, 7)
    a = identity_alignment(4, 7)
    assert torch.equal(aligned_cells(m), torch.tensor([m[a[t][0], t] for t in range(7)]))


def test_zero_copy_reads_aligned_source_layers(tiny_pair):
    src, _ = tiny_pair
    ids = torch.randint(2, 90, (1, 10))
    sk, sv = _capture(src, ids)                         # 2 source layers
    zc = ZeroCopyMapper(3, 16, 2, 1e6, n_src_layers=2)  # 3 target layers
    k, v = zc.map_content(sk, sv)
    assert len(k) == 3
    for lt, (ls,) in identity_alignment(2, 3).items():
        assert torch.allclose(k[lt], sk[ls]) and torch.allclose(v[lt], sv[ls])


def test_source_capture_uses_source_depth(tiny_pair):
    src, _ = tiny_pair
    k, v = source_prefill_content(src, torch.randint(2, 90, (1, 12)), prefix_len=8)
    assert len(k) == len(v) == 2
    with pytest.raises(ValueError, match="source has 2 layers"):
        source_prefill_content(src, torch.randint(2, 90, (1, 12)), 8, n_layers=3)


def _fitted_like(sel_k, sel_v, n_tgt, n_src, D=16, H=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    m = KVMapper(sel_k, D, H, 1e6, n_src_layers=n_src, src_layers_v=sel_v)
    for lt in range(n_tgt):
        for h in range(H):
            m.set("K", lt, h, torch.randn(len(sel_k[lt]) * D, D, generator=g) * 0.1, torch.zeros(D), 0.5)
            m.set("V", lt, h, torch.randn(len(sel_v[lt]) * D, D, generator=g) * 0.1, torch.zeros(D), 0.5)
    return m


def test_value_maps_read_their_own_selection():
    """Regression: V maps were applied to the KEY selection's layers."""
    D, H = 16, 2
    sel_k, sel_v = {0: [0]}, {0: [1]}
    m = KVMapper(sel_k, D, H, 1e6, n_src_layers=2, src_layers_v=sel_v)
    for h in range(H):
        m.set("K", 0, h, torch.eye(D), torch.zeros(D), 1.0)
        m.set("V", 0, h, torch.eye(D), torch.zeros(D), 1.0)
    sk = [torch.randn(1, H, 5, D) for _ in range(2)]
    sv = [torch.randn(1, H, 5, D) for _ in range(2)]
    k, v = m.map_content(sk, sv)
    assert torch.allclose(k[0], sk[0])
    assert torch.allclose(v[0], sv[1])


def test_state_dict_roundtrip_keeps_both_selections():
    m = _fitted_like({0: [0, 1], 1: [1, 0], 2: [1, 0]}, {0: [1, 0], 1: [0, 1], 2: [0, 1]}, 3, 2)
    r = KVMapper.from_state_dict(m.state_dict())
    assert r.src_layers == m.src_layers and r.src_layers_v == m.src_layers_v
    assert r.n_src_layers == 2
    x = [torch.randn(1, 2, 4, 16) for _ in range(2)]
    for a, b in zip(m.map_content(x, x), r.map_content(x, x)):
        for u, w in zip(a, b):
            assert torch.allclose(u, w)


def test_old_mappers_without_value_selection_warn():
    sd = _fitted_like({0: [0, 1]}, {0: [0, 1]}, 1, 2).state_dict()
    del sd["meta"]["src_layers_v"]
    with pytest.warns(UserWarning, match="Refit"):
        KVMapper.from_state_dict(sd)
    del sd["meta"]["n_src_layers"]
    single = _fitted_like({0: [0]}, {0: [0]}, 1, 2).state_dict()
    del single["meta"]["src_layers_v"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        KVMapper.from_state_dict(single)              # k=1 cannot be affected


def test_check_compat_rejects_wrong_pair():
    m = _fitted_like({0: [1], 1: [1], 2: [0]}, {0: [1], 1: [1], 2: [0]}, 3, 2)
    m.check_compat(2, 3, 2, 16)
    with pytest.raises(ValueError, match="target layers"):
        m.check_compat(2, 4, 2, 16)
    with pytest.raises(ValueError, match="source layers"):
        m.check_compat(3, 3, 2, 16)
    with pytest.raises(ValueError, match="shape"):
        m.check_compat(2, 3, 4, 16)
    del m.V[(2, 1)]
    with pytest.raises(ValueError, match="missing"):
        m.check_compat(2, 3, 2, 16)


def test_mapped_cache_runs_in_deeper_target(tiny_pair):
    """End to end: 2-layer source KV -> mapper -> 3-layer target decodes from it."""
    src, tgt = tiny_pair
    ids = torch.randint(2, 90, (2, 20))
    prefix = 12
    r2 = torch.rand(2, 3, 2)
    sel = top_k(r2, 2)
    m = _fitted_like(sel, sel, 3, 2)
    caches = [source_prefill_content(src, ids[i : i + 1], prefix) for i in range(2)]
    ppl = transfer_ppl(tgt, m, caches, ids, prefix)
    assert torch.isfinite(torch.tensor(ppl)) and ppl > 1
    zc = ZeroCopyMapper(3, 16, 2, 1e6, n_src_layers=2)
    assert torch.isfinite(torch.tensor(transfer_ppl(tgt, zc, caches, ids, prefix)))
    mean = MeanMapper([[torch.zeros(16)] * 2] * 3, [[torch.zeros(16)] * 2] * 3, 16, 2, 1e6, n_src_layers=2)
    assert torch.isfinite(torch.tensor(transfer_ppl(tgt, mean, caches, ids, prefix)))


def test_transfer_ppl_refuses_missing_caches(tiny_pair):
    src, tgt = tiny_pair
    ids = torch.randint(2, 90, (2, 16))
    caches = [source_prefill_content(src, ids[:1], 8)]
    zc = ZeroCopyMapper(3, 16, 2, 1e6, n_src_layers=2)
    with pytest.raises(ValueError, match="1 of 2"):
        transfer_ppl(tgt, zc, caches, ids, 8)


def test_mapper_runs_on_its_device():
    m = _fitted_like({0: [0]}, {0: [1]}, 1, 2)
    m.to("cpu")
    k, v = m.map_content([torch.randn(1, 2, 3, 16, dtype=torch.float16)] * 2,
                         [torch.randn(1, 2, 3, 16, dtype=torch.float16)] * 2)
    assert k[0].dtype == torch.float32 and k[0].device.type == "cpu"
