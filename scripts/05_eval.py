"""Phase 4: does a mapped cache actually work?

Prefix-conditioned perplexity on WikiText-2 (a different domain from the FineWeb-Edu
calibration set, so this measures transfer rather than memorization), against the four
baselines that make the number interpretable:

  target standalone  -- the ceiling; the target prefills its own prefix
  oracle graft       -- target's own cache pushed through the graft path; must match
                        the ceiling, else the harness is broken rather than the mapper
  zero-copy          -- source cache pasted in unmapped; "is the mapper doing anything?"
  mean prediction    -- per-channel means, the R^2 = 0 floor

Retention is reported as the ratio of log-perplexities rather than raw perplexity,
because perplexity is exponential in loss and raw ratios overstate small degradations.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from kvtransfer.capture import KVCapture
from kvtransfer.data import wikitext2
from kvtransfer.eval.ppl import oracle_ppl, standalone_ppl, transfer_ppl
from kvtransfer.mapper import KVMapper, MeanMapper, ZeroCopyMapper

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

_probe = import_module("02_probe_r2")
load_model, free = _probe.load_model, _probe.free

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def collect_source_caches(cfg, ids, prefix_len, n_layers, rt):
    """Prefill every eval prefix with the SOURCE model, keeping content-space KV.

    Done in one pass so the source model is loaded once and freed before the target
    model is needed -- the two never coexist.
    """
    model = load_model(cfg["source"], getattr(torch, rt["dtype"]), rt["device"])
    out = []
    try:
        for seq in ids:
            seq = seq.unsqueeze(0)[:, :prefix_len].to(rt["device"])
            with KVCapture(model, store_dtype=torch.float32) as cap:
                model(input_ids=seq, use_cache=False)
                out.append((
                    [cap.keys[l] for l in range(n_layers)],
                    [cap.values[l] for l in range(n_layers)],
                ))
    finally:
        free(model)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/qwen3_0.6b_to_1.7b.yaml"))
    ap.add_argument("--mappers", nargs="*", default=None, help="mapper .pt files")
    ap.add_argument("--n-seqs", type=int, default=None)
    ap.add_argument("--prefix-len", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "results/eval_ppl.json"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    rt, ev = cfg["runtime"], cfg["eval"]
    n_seqs = args.n_seqs or ev["wikitext_seqs"]
    prefix_len = args.prefix_len or ev["prefix_len"]

    from transformers import AutoConfig, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["target"])
    tcfg = AutoConfig.from_pretrained(cfg["target"])
    L, H, D = tcfg.num_hidden_layers, tcfg.num_key_value_heads, tcfg.head_dim
    theta = float(tcfg.rope_parameters["rope_theta"])

    # seq_len must leave room for a real continuation after the prefix, so it tracks
    # prefix_len rather than a fixed config default. A --prefix-len override (e.g. the
    # 4096-token length-generalization check) would otherwise silently score zero
    # continuation tokens against the config's fixed 1024-token sequences.
    min_continuation = ev.get("min_continuation", 256)
    seq_len = max(ev.get("seq_len", 1024), prefix_len + min_continuation)
    ids = wikitext2(tok, n_seqs=n_seqs, seq_len=seq_len)
    print(f"eval: {tuple(ids.shape)}, prefix_len={prefix_len}")

    print("prefilling prefixes with the source model ...")
    src_caches = collect_source_caches(cfg, ids, prefix_len, L, rt)

    mapper_paths = args.mappers or sorted((ROOT / "results/mappers").glob("*.pt"))
    mappers = {Path(p).stem: KVMapper.from_state_dict(torch.load(p)) for p in mapper_paths}
    mappers["zero_copy"] = ZeroCopyMapper(L, D, H, theta)

    # Mean baseline: per-(layer, head, channel) means of the TARGET's content-space KV,
    # taken from the mapper intercepts, which are exactly those means when W predicts
    # nothing. Using a fitted mapper's b avoids a second capture pass.
    if mapper_paths:
        ref = next(iter(mappers.values()))
        mk = [[ref.K[(l, h)].b for h in range(H)] for l in range(L)]
        mv = [[ref.V[(l, h)].b for h in range(H)] for l in range(L)]
        mappers["mean_pred"] = MeanMapper(mk, mv, D, H, theta)

    print("loading target model ...")
    model = load_model(cfg["target"], getattr(torch, rt["dtype"]), rt["device"])
    results = {}
    try:
        t = time.time()
        results["target_standalone"] = standalone_ppl(model, ids, prefix_len, rt["device"])
        print(f"  target standalone ppl = {results['target_standalone']:.3f}  ({time.time()-t:.0f}s)")

        t = time.time()
        results["oracle_graft"] = oracle_ppl(model, ids, prefix_len, L, rt["device"])
        print(f"  oracle graft      ppl = {results['oracle_graft']:.3f}  ({time.time()-t:.0f}s)")

        for name, mapper in mappers.items():
            t = time.time()
            results[name] = transfer_ppl(
                model, mapper, src_caches, ids, prefix_len, rt["device"]
            )
            print(f"  {name:18s} ppl = {results[name]:.3f}  ({time.time()-t:.0f}s)")
    finally:
        free(model)

    base = results["target_standalone"]
    import math

    print("\n=== retention (log-ppl ratio vs target standalone) ===")
    for k, v in results.items():
        ret = math.log(base) / math.log(v) * 100 if v > 1 else float("nan")
        print(f"  {k:18s} ppl {v:8.3f}   retention {ret:6.1f}%")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
