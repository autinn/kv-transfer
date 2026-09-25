"""Config-level plumbing shared by the probe, fit and eval scripts."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from .kvstore import KVStore, build_store
from .runtime import LazyModel, slug, store_dir


def load_config(path) -> dict:
    cfg = yaml.safe_load(open(path))
    for key in ("source", "target", "calibration", "runtime"):
        if key not in cfg:
            raise KeyError(f"{path}: missing required key '{key}'")
    return cfg


def calibration_ids(cfg: dict, tok, n_seqs: int, root: Path) -> torch.Tensor:
    """(n_seqs, seq_len) token ids, tokenized once and cached."""
    from .data import cached_calibration

    cal = cfg["calibration"]
    if cal.get("dataset", "fineweb-edu") == "synthetic":
        # Random ids: exercises every code path offline (tests, smoke runs). Not for results.
        g = torch.Generator().manual_seed(cal.get("seed", 0))
        return torch.randint(0, cal.get("vocab", 1000), (n_seqs, cal["seq_len"]), generator=g)
    return cached_calibration(
        tok, n_seqs, cal["seq_len"], cal["seed"],
        root / f"results/calib_{n_seqs}x{cal['seq_len']}_s{cal['seed']}.pt",
    )


def capture_both(cfg: dict, rt: dict, items: list[torch.Tensor], tag: str, root: Path,
                 stride: int = 1) -> tuple[KVStore, KVStore]:
    """Stores for the same token set from both models, loading each model at most once."""
    from transformers import AutoConfig

    stores = []
    for name in (cfg["source"], cfg["target"]):
        lazy = LazyModel(name, rt)
        try:
            stores.append(build_store(
                lazy.get, name, AutoConfig.from_pretrained(name), items,
                store_dir(cfg, root) / slug(name) / tag,
                stride=stride, shard_size=rt.get("chunk", 16),
                batch_size=rt.get("batch_size", 1),
                store_dtype=rt.get("store_dtype", "float16"),
            ))
        finally:
            lazy.release()
    return stores[0], stores[1]


def calibration_stores(cfg: dict, rt: dict, ids: torch.Tensor, root: Path):
    cal = cfg["calibration"]
    tag = f"calib_{ids.shape[0]}x{ids.shape[1]}_s{cal.get('seed', 0)}_st{cal['stride']}"
    return capture_both(cfg, rt, list(ids), tag, root, stride=cal["stride"])


def paired_batches(src: KVStore, tgt: KVStore):
    """Zip two calibration stores shard by shard, checking the tokens line up."""
    if len(src) != len(tgt) or src.meta["n_shards"] != tgt.meta["n_shards"]:
        raise ValueError("source and target stores were captured from different token sets")
    for (sk, sv), (tk, tv) in zip(src.iter_batches(), tgt.iter_batches()):
        if sk.shape[1] != tk.shape[1] or sk.shape[3] != tk.shape[3]:
            raise ValueError(f"shard shape mismatch: source {tuple(sk.shape)} vs target {tuple(tk.shape)}")
        yield sk, sv, tk, tv
