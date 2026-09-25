"""Capture content-space KV once, to disk, and stream it back for every later pass.

Before this, every stage reloaded both models for every chunk of `runtime.chunk`
sequences, and `03_fit.py` repeated the whole capture for every k in the sweep. That
was tolerable for 0.6B/1.7B. For 14B -> 32B it means loading 64 GB of weights about a
hundred times per pass, ten passes over.

Now each model is loaded once per token set. Its pre-RoPE keys and values are written
as shards, and the probe, every top-k fit and the evaluations read those shards
instead of running the model again.

    <root>/manifest.json
    <root>/shard_00000.pt   {"k": [(L, H, S_i, D), ...], "v": [...]}   one entry per item

Items may differ in length (benchmark contexts do). Calibration items share a length,
so `iter_batches` stacks each shard back into (L, B, H, S, D).

A store is keyed by a fingerprint of (model, token ids, stride, dtype). A matching
complete store is reused without loading the model. A matching partial store resumes
at the first missing shard. A mismatched one is cleared.

Rotated keys are not stored: they are recomputed from content keys at the stored
positions (`rope_keys`), which halves disk use.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator

import torch

from .capture import KVCapture
from .rope import apply_rope, cos_sin
from .runtime import input_device, kv_dims, rope_theta
from .stream import subsample

FORMAT_VERSION = 1


def fingerprint(model_name: str, items: list[torch.Tensor], stride: int, dtype: str) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(
        {"model": model_name, "stride": stride, "dtype": dtype, "format": FORMAT_VERSION},
        sort_keys=True,
    ).encode())
    for t in items:
        h.update(t.to(torch.int64).contiguous().numpy().tobytes())
        h.update(b"|")
    return h.hexdigest()[:20]


def _atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


class KVStore:
    """Read side of a captured store."""

    def __init__(self, root):
        self.root = Path(root)

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def manifest(self) -> dict | None:
        if not self.manifest_path.exists():
            return None
        try:
            return json.loads(self.manifest_path.read_text())
        except json.JSONDecodeError:
            return None          # a torn write; treat as absent and recapture

    def is_complete(self, fp: str | None = None) -> bool:
        m = self.manifest()
        return bool(m and m.get("complete") and (fp is None or m.get("fingerprint") == fp))

    @property
    def meta(self) -> dict:
        m = self.manifest()
        if not m or not m.get("complete"):
            raise RuntimeError(f"KV store at {self.root} is missing or incomplete")
        return m

    def __len__(self) -> int:
        return self.meta["n_items"]

    def shard_path(self, i: int) -> Path:
        return self.root / f"shard_{i:05d}.pt"

    def iter_shards(self) -> Iterator[tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """Yield (keys, values) per shard, each a list of (L, H, S_i, D) tensors."""
        for i in range(self.meta["n_shards"]):
            d = torch.load(self.shard_path(i), map_location="cpu", weights_only=True, mmap=True)
            yield d["k"], d["v"]

    def iter_items(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield (k, v) per item, each (L, H, S_i, D)."""
        for ks, vs in self.iter_shards():
            yield from zip(ks, vs)

    def iter_batches(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield (k, v) per shard stacked to (L, B, H, S, D). Items must share a length."""
        for ks, vs in self.iter_shards():
            if len({k.shape for k in ks}) != 1:
                raise ValueError("iter_batches needs equal-length items; use iter_items")
            yield torch.stack(ks, dim=1), torch.stack(vs, dim=1)

    def positions(self, n_stored: int, device=None) -> torch.Tensor:
        """Original token positions of the first `n_stored` stored tokens."""
        return torch.arange(n_stored, device=device) * self.meta["stride"]


def rope_keys(k: torch.Tensor, store: KVStore, dtype=torch.float32) -> torch.Tensor:
    """Rotate stored content keys (..., S, D) to their original positions."""
    m = store.meta
    pos = store.positions(k.shape[-2], device=k.device)
    cos, sin = cos_sin(pos, m["head_dim"], m["rope_theta"], dtype=torch.float32)
    return apply_rope(k.to(torch.float32), cos, sin).to(dtype)


def estimate_bytes(items: list[torch.Tensor], stride: int, L: int, H: int, D: int, dtype) -> int:
    per_tok = 2 * L * H * D * torch.empty((), dtype=dtype).element_size()
    return sum(math.ceil(len(t) / stride) for t in items) * per_tok


@torch.no_grad()
def build_store(
    get_model: Callable[[], torch.nn.Module],
    model_name: str,
    model_cfg,
    items: Iterable[torch.Tensor],
    root,
    stride: int = 1,
    shard_size: int = 16,
    batch_size: int = 1,
    store_dtype: str = "float16",
    log: Callable[[str], None] = print,
) -> KVStore:
    """Capture pre-RoPE KV for `items` (1-D token tensors) into a store at `root`.

    `get_model` is called only if something actually needs capturing, so a stage whose
    store is already complete never loads the model. It is not freed here: the caller
    may capture several token sets with one load.
    """
    items = [torch.as_tensor(t, dtype=torch.long).flatten() for t in items]
    if not items:
        raise ValueError("no items to capture")
    if min(len(t) for t in items) < 1:
        raise ValueError("cannot capture an empty item")
    root = Path(root)
    fp = fingerprint(model_name, items, stride, store_dtype)
    store = KVStore(root)
    if store.is_complete(fp):
        log(f"  reusing KV store {root} ({len(items)} items)")
        return store

    m = store.manifest()
    if m is None or m.get("fingerprint") != fp:
        if root.exists():
            log(f"  clearing stale KV store {root}")
            shutil.rmtree(root)
        m = None
    root.mkdir(parents=True, exist_ok=True)

    L, H, D = kv_dims(model_cfg)
    sdt = getattr(torch, store_dtype)
    need = estimate_bytes(items, stride, L, H, D, sdt)
    free_bytes = shutil.disk_usage(root).free
    if need > free_bytes * 0.95:
        raise RuntimeError(
            f"KV store {root} needs ~{need/1e9:.1f} GB but only {free_bytes/1e9:.1f} GB is free. "
            f"Point runtime.store_dir at a bigger disk or lower the item count."
        )

    n_shards = math.ceil(len(items) / shard_size)
    manifest = {
        "format": FORMAT_VERSION, "fingerprint": fp, "model": model_name,
        "n_items": len(items), "n_shards": n_shards, "shard_size": shard_size,
        "stride": stride, "dtype": store_dtype, "n_layers": L, "kv_heads": H,
        "head_dim": D, "rope_theta": rope_theta(model_cfg),
        "done": sorted(set(m["done"])) if m else [], "complete": False,
    }
    _atomic_json(store.manifest_path, manifest)
    done = set(manifest["done"])
    todo = [i for i in range(n_shards) if i not in done or not store.shard_path(i).exists()]
    log(f"  capturing {model_name} -> {root}: {len(todo)}/{n_shards} shards, ~{need/1e9:.1f} GB")

    model = get_model()
    dev = input_device(model)
    t0 = time.time()
    for n, si in enumerate(todo):
        chunk = items[si * shard_size : (si + 1) * shard_size]
        ks, vs = [], []
        j = 0
        while j < len(chunk):
            # Batch consecutive equal-length items; variable-length ones go one at a time.
            b = [chunk[j]]
            while len(b) < batch_size and j + len(b) < len(chunk) and len(chunk[j + len(b)]) == len(b[0]):
                b.append(chunk[j + len(b)])
            j += len(b)
            with KVCapture(model, store_dtype=torch.float32) as cap:
                model(input_ids=torch.stack(b).to(dev), use_cache=False)
                k, v = cap.stacked()                     # (L, B, H, S, D) fp32 on CPU
            k, v = subsample(k, stride), subsample(v, stride)
            if not (torch.isfinite(k).all() and torch.isfinite(v).all()):
                raise FloatingPointError(
                    f"non-finite KV captured from {model_name}; run the model in "
                    f"bfloat16 or float32 instead of {model.dtype}"
                )
            k, v = k.to(sdt), v.to(sdt)
            if not (torch.isfinite(k).all() and torch.isfinite(v).all()):
                raise FloatingPointError(f"KV overflows {store_dtype}; set runtime.store_dtype: bfloat16")
            for i in range(len(b)):
                ks.append(k[:, i].clone())
                vs.append(v[:, i].clone())
            cap.clear()
        path = store.shard_path(si)
        tmp = path.with_suffix(".pt.tmp")
        torch.save({"k": ks, "v": vs}, tmp)
        os.replace(tmp, path)
        done.add(si)
        manifest["done"] = sorted(done)
        _atomic_json(store.manifest_path, manifest)
        el = time.time() - t0
        log(f"    shard {n+1}/{len(todo)}  {el/60:.1f}m elapsed, "
            f"~{el/(n+1)*(len(todo)-n-1)/60:.0f}m left")

    manifest["complete"] = True
    _atomic_json(store.manifest_path, manifest)
    return store


def as_layer_list(k: torch.Tensor) -> list[torch.Tensor]:
    """(L, H, S, D) item -> per-layer list of (1, H, S, D), the shape mappers consume."""
    return [k[l].unsqueeze(0) for l in range(k.shape[0])]
