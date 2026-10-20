"""Transfer vs re-prefill latency: the paper's speedup claim (2.7-25x, its Tables 5 and 16).

Protocol, following the paper's Appendix G: synthetic prompts at each length, bf16
models, warmup then timed trials per cell (paper: 50 / 30), re-prefill = the target's
transformer body without the LM head, transfer = mapping an existing source cache into
the target's format. The mapper has random weights of the fitted shape (see bench.py),
so no fit is needed. Differences from the paper are printed with the results.

Three timings per length:
  prefill   target body over the prompt                       (what transfer replaces)
  transfer  source cache -> target cache, per mapper dtype    (the paper's speedup)
  e2e       source prefill + transfer                         (not in the paper)

    python scripts/10_speedup.py --config configs/qwen3_0.6b_to_1.7b.a10.yaml
    python scripts/10_speedup.py --config C --lengths 64 1024 --warmup 3 --trials 5   # quick

Writes <output_dir>/speedup_<gpu>.json and prints a table.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

from kvtransfer.bench import (
    body_params, mapper_flops, prefill, prefill_flops, random_mapper, synthetic_cache,
    time_ms, transfer,
)
from kvtransfer.pipeline import load_config
from kvtransfer.runtime import free, kv_dims, load_model, output_dir, pair_dims, rope_theta

ROOT = Path(__file__).resolve().parent.parent
PAPER_LENGTHS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

# Paper Table 5 / Table 16 (8x H100, bf16, FlashAttention-2), for printing beside ours.
PAPER = {
    ("Qwen/Qwen3-14B", "Qwen/Qwen3-32B"): {
        "k": 8, "cells": {64: (14.0, 61.7), 8192: (67.8, 1154.8), 32768: (277.6, 6975.3)},
        "range": "4.4-25.1x"},
    ("Qwen/Qwen3-8B", "Qwen/Qwen3-32B"): {"k": 12, "cells": {}, "range": "4.3-17.8x"},
}
PAPER_NOTE = ("paper's smallest pair is Ministral 3B -> 8B: 2.7-4.0x over 64 to 32K tokens; "
              "its Qwen3 pairs (14B/8B -> 32B): 4.3-25.1x")


def is_oom(e: Exception) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def run_cells(name, lengths, make_fn, args, device, results):
    """Time make_fn(S)() at each length; after an OOM, skip the longer lengths."""
    for S in lengths:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            fn = make_fn(S)
            r = time_ms(fn, args.warmup, args.trials, device)
            if torch.cuda.is_available():
                r["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
            del fn
        except Exception as e:  # noqa: BLE001 -- only OOM is expected and recorded
            if not is_oom(e):
                raise
            print(f"  {name} S={S}: out of memory, skipping longer lengths", flush=True)
            results.setdefault(name, {})[S] = None
            for S2 in [x for x in lengths if x > S]:
                results[name][S2] = None
            return
        results.setdefault(name, {})[S] = r
        print(f"  {name:<16} S={S:>6}  {r['median']:9.2f} ms (median of {r['n']})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=PAPER_LENGTHS)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--same-head", action="store_true", help="same-head features (default all-heads)")
    ap.add_argument("--mapper-dtypes", nargs="+", default=["float32", "bfloat16"])
    ap.add_argument("--attn", default="sdpa",
                    help="attention kernel for both models (paper: flash_attention_2)")
    ap.add_argument("--no-e2e", action="store_true", help="skip loading the source model")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from transformers import AutoConfig

    cfg = load_config(args.config)
    src_name, tgt_name = cfg["source"], cfg["target"]
    src_cfg, tgt_cfg = AutoConfig.from_pretrained(src_name), AutoConfig.from_pretrained(tgt_name)
    Ls, Lt, H, D = pair_dims(src_cfg, tgt_cfg)
    th_s, th_t = rope_theta(src_cfg), rope_theta(tgt_cfg)
    dev = args.device
    model_dtype = torch.bfloat16 if dev != "cpu" else torch.float32
    lengths = sorted(args.lengths)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor() or "cpu"
    print(f"{src_name} -> {tgt_name}  |  {gpu}  |  k={args.k} "
          f"{'same-head' if args.same_head else 'all-heads'}  |  warmup {args.warmup}, "
          f"trials {args.trials}  |  attn {args.attn}", flush=True)
    vocab = tgt_cfg.vocab_size
    ids = {S: torch.randint(0, vocab, (1, S), device=dev) for S in lengths}
    res: dict = {}
    flops: dict = {}

    print("\n=== re-prefill: target body ===", flush=True)
    tgt = load_model(tgt_name, model_dtype, dev, attn_implementation=args.attn)
    n_body_t = body_params(tgt)
    for S in lengths:
        flops.setdefault(S, {})["prefill"] = prefill_flops(
            n_body_t, Lt, tgt_cfg.num_attention_heads, D, S)
    run_cells("prefill", lengths, lambda S: (lambda: prefill(tgt, ids[S])), args, dev, res)
    free(tgt)
    del tgt

    print("\n=== transfer: source cache -> target cache ===", flush=True)
    mappers = {}
    for name in args.mapper_dtypes:
        m = random_mapper(Ls, Lt, H, D, th_t, args.k, all_heads=not args.same_head, device=dev)
        m.to(dev, getattr(torch, name))
        mappers[name] = m
    for S in lengths:
        flops[S]["mapper"] = mapper_flops(next(iter(mappers.values())), S)
    src_caches = {}

    def src_cache(S):
        if S not in src_caches:
            src_caches.clear()  # keep one length resident
            src_caches[S] = synthetic_cache(Ls, H, D, S, th_s, dtype=model_dtype, device=dev)
        return src_caches[S]

    for name, m in mappers.items():
        run_cells(f"transfer_{name}", lengths,
                  lambda S, m=m: (lambda c=src_cache(S): transfer(m, c, D, th_s, model_dtype, tgt_cfg)),
                  args, dev, res)
    src_caches.clear()

    if not args.no_e2e:
        print("\n=== end to end: source prefill + transfer ===", flush=True)
        src = load_model(src_name, model_dtype, dev, attn_implementation=args.attn)
        n_body_s = body_params(src)
        for S in lengths:
            flops[S]["source_prefill"] = prefill_flops(
                n_body_s, Ls, src_cfg.num_attention_heads, D, S)
        run_cells("source_prefill", lengths, lambda S: (lambda: prefill(src, ids[S])), args, dev, res)
        for name, m in mappers.items():
            run_cells(f"e2e_{name}", lengths,
                      lambda S, m=m: (lambda: transfer(m, prefill(src, ids[S]), D, th_s,
                                                       model_dtype, tgt_cfg)),
                      args, dev, res)
        free(src)

    report(res, flops, lengths, args, src_name, tgt_name, gpu, cfg)


def med(res, name, S):
    r = res.get(name, {}).get(S)
    return None if r is None else r["median"]


SHORT = {"float32": "fp32", "bfloat16": "bf16", "float16": "fp16"}


def report(res, flops, lengths, args, src_name, tgt_name, gpu, cfg):
    dtypes = args.mapper_dtypes
    print("\n=== speedup = re-prefill / transfer (median ms) ===")
    head = f"{'tokens':>7} {'prefill':>9}"
    for d in dtypes:
        head += f" {'xfer ' + SHORT.get(d, d):>11} {'speedup':>8}"
    head += f" {'FLOP ratio':>10}"
    if not args.no_e2e:
        head += f" {'e2e ' + SHORT.get(dtypes[-1], dtypes[-1]):>10} {'e2e x':>6}"
    print(head)
    rows = []
    for S in lengths:
        p = med(res, "prefill", S)
        line, row = f"{S:>7} {p if p is None else round(p, 2)!s:>9}", {"tokens": S, "prefill_ms": p}
        for d in dtypes:
            t = med(res, f"transfer_{d}", S)
            sp = p / t if p and t else None
            row[f"transfer_{d}_ms"], row[f"speedup_{d}"] = t, sp
            line += f" {t if t is None else round(t, 2)!s:>11} {'' if sp is None else f'{sp:.1f}x':>8}"
        fr = flops[S]["prefill"] / flops[S]["mapper"]
        row["flop_ratio"] = fr
        line += f" {fr:>9.1f}x"
        if not args.no_e2e:
            e = med(res, f"e2e_{dtypes[-1]}", S)
            ex = p / e if p and e else None
            row[f"e2e_{dtypes[-1]}_ms"], row["e2e_speedup"] = e, ex
            line += f" {e if e is None else round(e, 2)!s:>10} {'' if ex is None else f'{ex:.2f}x':>6}"
        print(line)
        rows.append(row)

    paper = PAPER.get((src_name, tgt_name))
    print()
    if paper:
        print(f"paper (k={paper['k']}, 8x H100, FA2): speedup {paper['range']}")
        for S, (t, p) in paper["cells"].items():
            print(f"  {S:>6} tokens: transfer {t} ms, re-prefill {p} ms, {p / t:.1f}x")
    else:
        print(f"no such pair in the paper; {PAPER_NOTE}")
    print("FLOP ratio = re-prefill FLOPs / mapper FLOPs: the speedup if both ran at the same "
          "efficiency.\nMeasured below it means fixed overhead (Python dispatch, small kernels) "
          "dominates.")
    print("Differences from the paper: one GPU (no cross-GPU transfer, which the paper "
          f"includes); attention kernel {args.attn}; random-weight mapper (same arithmetic).")

    out = output_dir(cfg, ROOT) / f"speedup_{gpu.replace(' ', '_').replace('/', '_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "source": src_name, "target": tgt_name, "gpu": gpu, "k": args.k,
        "all_heads": not args.same_head, "attn": args.attn, "warmup": args.warmup,
        "trials": args.trials, "torch": torch.__version__, "rows": rows,
        "raw": {n: {str(S): r for S, r in v.items()} for n, v in res.items()},
        "flops": {str(S): f for S, f in flops.items()},
    }, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
