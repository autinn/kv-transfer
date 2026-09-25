"""Phase 4: does a mapped cache actually work?

Prefix-conditioned perplexity on WikiText-2 (a different domain from the FineWeb-Edu
calibration set, so this measures transfer rather than memorization), against the four
baselines that make the number interpretable:

  target standalone  -- the ceiling; the target prefills its own prefix
  oracle graft       -- target's own cache pushed through the graft path; must match
                        the ceiling, else the harness is broken rather than the mapper
  zero-copy          -- source cache pasted in unmapped; "is the mapper doing anything?"
                        (layer-aligned proportionally when the depths differ)
  mean prediction    -- per-channel means, the R^2 = 0 floor

Retention is reported as the ratio of log-perplexities rather than raw perplexity,
because perplexity is exponential in loss and raw ratios overstate small degradations.

The source model prefills every prefix once into a disk store and is freed before the
target loads, so the two are never resident together.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from kvtransfer.data import wikitext2
from kvtransfer.eval.ppl import oracle_ppl, standalone_ppl, transfer_ppl
from kvtransfer.kvstore import KVStore, as_layer_list, build_store
from kvtransfer.mapper import KVMapper, MeanMapper, ZeroCopyMapper
from kvtransfer.pipeline import load_config
from kvtransfer.runtime import (
    LazyModel, input_device, output_dir, pair_dims, resolve_runtime, rope_theta, slug, store_dir,
)

ROOT = Path(__file__).resolve().parents[1]


def eval_ids(cfg, tok, n_seqs, seq_len) -> torch.Tensor:
    ev = cfg.get("eval", {})
    if ev.get("dataset", "wikitext2") == "synthetic":
        g = torch.Generator().manual_seed(1)
        return torch.randint(0, cfg["calibration"].get("vocab", 1000), (n_seqs, seq_len), generator=g)
    return wikitext2(tok, n_seqs=n_seqs, seq_len=seq_len)


def source_prefix_store(cfg, rt, ids, prefix_len) -> KVStore:
    """Prefill every eval prefix with the SOURCE model into a disk store."""
    from transformers import AutoConfig

    lazy = LazyModel(cfg["source"], rt)
    try:
        return build_store(
            lazy.get, cfg["source"], AutoConfig.from_pretrained(cfg["source"]),
            list(ids[:, :prefix_len]),
            store_dir(cfg, ROOT) / slug(cfg["source"]) / f"ppl_{ids.shape[0]}x{ids.shape[1]}_p{prefix_len}",
            stride=1, shard_size=rt.get("chunk", 16),
            store_dtype=rt.get("store_dtype", "float16"),
        )
    finally:
        lazy.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/qwen3_0.6b_to_1.7b.yaml"))
    ap.add_argument("--mappers", nargs="*", default=None, help="default: <output_dir>/mappers/*.pt")
    ap.add_argument("--n-seqs", type=int, default=None)
    ap.add_argument("--prefix-len", type=int, default=None)
    ap.add_argument("--device", default=None, help="override runtime.device")
    ap.add_argument("--out", default=None, help="default: <output_dir>/eval_ppl.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ev = cfg.get("eval", {})
    rt = resolve_runtime(cfg["runtime"], args.device)
    n_seqs = args.n_seqs or ev.get("wikitext_seqs", 64)
    prefix_len = args.prefix_len or ev.get("prefix_len", 512)
    out_root = output_dir(cfg, ROOT)

    from transformers import AutoConfig, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["target"])
    tcfg = AutoConfig.from_pretrained(cfg["target"])
    Ls, Lt, H, D = pair_dims(AutoConfig.from_pretrained(cfg["source"]), tcfg)
    theta = rope_theta(tcfg)

    # seq_len must leave room for a real continuation after the prefix, so it tracks
    # prefix_len rather than a fixed config default. A --prefix-len override (e.g. the
    # 4096-token length-generalization check) would otherwise silently score zero
    # continuation tokens against the config's fixed 1024-token sequences.
    min_continuation = ev.get("min_continuation", 256)
    seq_len = max(ev.get("seq_len", 1024), prefix_len + min_continuation)
    ids = eval_ids(cfg, tok, n_seqs, seq_len)
    print(f"eval: {tuple(ids.shape)}, prefix_len={prefix_len} | {Ls} -> {Lt} layers")

    mapper_paths = args.mappers if args.mappers is not None else sorted((out_root / "mappers").glob("*.pt"))
    mappers = {}
    for p in mapper_paths:
        m = KVMapper.from_state_dict(torch.load(p))
        m.check_compat(Ls, Lt, H, D)          # fail now, not after an hour of prefill
        mappers[Path(p).stem] = m
    mappers["zero_copy"] = ZeroCopyMapper(Lt, D, H, theta, n_src_layers=Ls)

    # Mean baseline: per-(layer, head, channel) means of the TARGET's content-space KV,
    # taken from the mapper intercepts, which are exactly those means when W predicts
    # nothing. Using a fitted mapper's b avoids a second capture pass.
    fitted = [m for name, m in mappers.items() if name != "zero_copy"]
    if fitted:
        ref = fitted[0]
        mk = [[ref.K[(l, h)].b for h in range(H)] for l in range(Lt)]
        mv = [[ref.V[(l, h)].b for h in range(H)] for l in range(Lt)]
        mappers["mean_pred"] = MeanMapper(mk, mv, D, H, theta, n_src_layers=Ls)

    print("prefilling prefixes with the source model ...")
    store = source_prefix_store(cfg, rt, ids, prefix_len)

    print("loading target model ...")
    lazy = LazyModel(cfg["target"], rt)
    model = lazy.get()
    dev = input_device(model)
    results = {}
    try:
        t = time.time()
        results["target_standalone"] = standalone_ppl(model, ids, prefix_len, dev)
        print(f"  target standalone ppl = {results['target_standalone']:.3f}  ({time.time()-t:.0f}s)")

        t = time.time()
        results["oracle_graft"] = oracle_ppl(model, ids, prefix_len, Lt, dev)
        print(f"  oracle graft      ppl = {results['oracle_graft']:.3f}  ({time.time()-t:.0f}s)")

        for name, mapper in mappers.items():
            t = time.time()
            mapper.to(dev)
            src_caches = ((as_layer_list(k), as_layer_list(v)) for k, v in store.iter_items())
            results[name] = transfer_ppl(model, mapper, src_caches, ids, prefix_len, dev)
            mapper.to("cpu")
            print(f"  {name:18s} ppl = {results[name]:.3f}  ({time.time()-t:.0f}s)")
    finally:
        lazy.release()

    base = results["target_standalone"]
    print("\n=== retention (log-ppl ratio vs target standalone) ===")
    retention = {}
    for k, v in results.items():
        ret = math.log(base) / math.log(v) * 100 if v > 1 else float("nan")
        retention[k] = ret
        print(f"  {k:18s} ppl {v:8.3f}   retention {ret:6.1f}%")
    # Kept apart from the per-config perplexities so readers of those are unaffected.
    results["retention_pct"] = retention

    out = Path(args.out) if args.out else out_root / "eval_ppl.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
