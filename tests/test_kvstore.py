"""Disk-backed KV store: capture once, reuse, resume, and never serve stale data."""

import json

import pytest
import torch

from kvtransfer.capture import KVCapture
from kvtransfer.kvstore import KVStore, build_store, rope_keys
from kvtransfer.rope import apply_rope, rope_tables_for
from kvtransfer.stream import subsample


def _items(n=5, S=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    return list(torch.randint(2, 90, (n, S), generator=g))


class Loader:
    def __init__(self, model):
        self.model, self.calls = model, 0

    def __call__(self):
        self.calls += 1
        return self.model


def _build(model, items, root, **kw):
    loader = Loader(model)
    kw = {"stride": 2, "shard_size": 2, **kw}
    store = build_store(loader, "tiny", model.config, items, root, log=lambda *_: None, **kw)
    return store, loader


def test_roundtrip_matches_direct_capture(tiny_pair, tmp_path):
    src, _ = tiny_pair
    items = _items()
    store, _ = _build(src, items, tmp_path / "s", batch_size=2)
    assert len(store) == len(items) and store.meta["n_shards"] == 3
    for item, (k, v) in zip(items, store.iter_items()):
        with torch.no_grad(), KVCapture(src, store_dtype=torch.float32) as cap:
            src(input_ids=item[None], use_cache=False)
            rk, rv = cap.stacked()
        rk, rv = subsample(rk, 2)[:, 0], subsample(rv, 2)[:, 0]
        assert k.shape == rk.shape
        assert torch.allclose(k.float(), rk, atol=2e-3, rtol=1e-3)
        assert torch.allclose(v.float(), rv, atol=2e-3, rtol=1e-3)


def test_complete_store_is_reused_without_loading(tiny_pair, tmp_path):
    src, _ = tiny_pair
    items = _items()
    _, first = _build(src, items, tmp_path / "s")
    _, second = _build(src, items, tmp_path / "s")
    assert first.calls == 1 and second.calls == 0


def test_changed_tokens_invalidate_store(tiny_pair, tmp_path):
    src, _ = tiny_pair
    _build(src, _items(seed=0), tmp_path / "s")
    store, loader = _build(src, _items(seed=1), tmp_path / "s")
    assert loader.calls == 1
    first = next(store.iter_items())[0]
    with torch.no_grad(), KVCapture(src, store_dtype=torch.float32) as cap:
        src(input_ids=_items(seed=1)[0][None], use_cache=False)
        ref = subsample(cap.stacked()[0], 2)[:, 0]
    assert torch.allclose(first.float(), ref, atol=2e-3, rtol=1e-3)


def test_partial_store_resumes_missing_shards_only(tiny_pair, tmp_path):
    src, _ = tiny_pair
    items = _items()
    store, _ = _build(src, items, tmp_path / "s")
    before = [k.clone() for k, _ in store.iter_items()]
    m = json.loads(store.manifest_path.read_text())
    m["complete"], m["done"] = False, [0, 1]
    store.manifest_path.write_text(json.dumps(m))
    mtime0 = store.shard_path(0).stat().st_mtime_ns
    store.shard_path(2).unlink()

    store, loader = _build(src, items, tmp_path / "s")
    assert loader.calls == 1 and store.is_complete()
    assert store.shard_path(0).stat().st_mtime_ns == mtime0      # not rewritten
    for a, (b, _) in zip(before, store.iter_items()):
        assert torch.equal(a, b)


def test_variable_length_items(tiny_pair, tmp_path):
    src, _ = tiny_pair
    items = [torch.arange(2, 2 + n) for n in (3, 7, 5)]
    store, _ = _build(src, items, tmp_path / "s", stride=1, batch_size=4)
    assert [k.shape[2] for k, _ in store.iter_items()] == [3, 7, 5]
    with pytest.raises(ValueError, match="equal-length"):
        next(store.iter_batches())


def test_iter_batches_restacks_layer_batch_layout(tiny_pair, tmp_path):
    src, _ = tiny_pair
    store, _ = _build(src, _items(n=4), tmp_path / "s")
    k, v = next(store.iter_batches())
    L, H, D = 2, 2, 16
    assert k.shape == (L, 2, H, 12, D) and v.shape == k.shape


def test_rope_keys_match_rotating_before_subsampling(tiny_pair, tmp_path):
    """Stored keys are content-space; rotating them at stride positions must equal the
    old path of rotating the full sequence and then subsampling."""
    src, _ = tiny_pair
    item = _items(n=1)[0]
    store, _ = _build(src, [item], tmp_path / "s", store_dtype="float32")
    k = next(store.iter_items())[0]
    with torch.no_grad(), KVCapture(src, store_dtype=torch.float32) as cap:
        src(input_ids=item[None], use_cache=False)
        full = cap.stacked()[0][:, 0]
    cos, sin = rope_tables_for(full.shape[2], 16, 1e6)
    ref = subsample(apply_rope(full, cos[0], sin[0]), 2, seq_dim=2)
    assert torch.allclose(rope_keys(k, store), ref, atol=1e-5)


def test_refuses_when_disk_is_too_small(tiny_pair, tmp_path, monkeypatch):
    import shutil
    from collections import namedtuple

    src, _ = tiny_pair
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: Usage(10, 10, 10))
    with pytest.raises(RuntimeError, match="GB is free"):
        _build(src, _items(), tmp_path / "s")


def test_torn_manifest_is_treated_as_absent(tmp_path):
    (tmp_path / "manifest.json").write_text("{not json")
    assert KVStore(tmp_path).manifest() is None
    assert not KVStore(tmp_path).is_complete()


def test_padded_batches_match_one_item_at_a_time(tiny_pair, tmp_path):
    """Variable-length items batched with right-padding store the same KV as batch 1."""
    src, _ = tiny_pair
    g = torch.Generator().manual_seed(1)
    items = [torch.randint(2, 90, (n,), generator=g) for n in (3, 11, 7, 7, 2, 9)]
    for stride in (1, 2):
        a, _ = _build(src, items, tmp_path / f"a{stride}", stride=stride, shard_size=4,
                      batch_size=1, store_dtype="float32")
        b, _ = _build(src, items, tmp_path / f"b{stride}", stride=stride, shard_size=4,
                      batch_size=4, store_dtype="float32")
        for (ka, va), (kb, vb) in zip(a.iter_items(), b.iter_items()):
            assert ka.shape == kb.shape
            assert torch.allclose(ka, kb, atol=1e-5) and torch.allclose(va, vb, atol=1e-5)
