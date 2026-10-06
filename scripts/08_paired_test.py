"""Paired significance tests between configurations on the same benchmark questions.

Reads the raw per-context scores 07_eval_tasks.py saves (eval_tasks.partial.json),
rebuilds per-question correctness exactly as the reported metric does
(`harness.mc_correct`), and for each pair of configurations reports:

  - the accuracy difference and a paired bootstrap 95% CI (questions resampled jointly),
  - McNemar's exact test on the discordant questions (one right, the other wrong).

Unpaired standard errors (~0.5 points on HellaSwag) overstate the noise in a difference
between two configurations scored on the same questions: most questions are right or wrong
under both, and only the discordant ones carry information about the gap.

Only compare configurations scored in the same run environment: raw scores from different
hardware differ slightly even for identical models (REPORT3's standalone 0.6035 on an A10
vs 0.6031 on an L4), which a paired test would wrongly attribute to the configuration.

No model is loaded; only the target tokenizer, to rebuild the task's request structure.
"""

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

from kvtransfer.eval.harness import encode_task, mc_correct
from kvtransfer.eval.tasks import load_task
from kvtransfer.pipeline import load_config
from kvtransfer.runtime import output_dir

ROOT = Path(__file__).resolve().parents[1]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: discordant counts b (A right, B wrong), c (reverse)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    """95% percentile CI of mean(a) - mean(b), resampling questions jointly."""
    rng = np.random.default_rng(seed)
    diff = a.astype(np.float64) - b.astype(np.float64)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/qwen3_0.6b_to_1.7b.a10.yaml"))
    ap.add_argument("--raw", default=None, help="default: <output_dir>/eval_tasks.partial.json")
    ap.add_argument("--task", default="hellaswag")
    ap.add_argument("--metric", default=None, help="default: acc_norm for hellaswag/arc, else acc")
    ap.add_argument("--configs", nargs="*", default=None,
                    help="configurations to compare pairwise (default: every one in the raw file)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw_path = Path(args.raw) if args.raw else output_dir(cfg, ROOT) / "eval_tasks.partial.json"
    partial = json.loads(raw_path.read_text())
    metric = args.metric or ("acc_norm" if args.task in ("hellaswag", "arc_challenge") else "acc")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["target"])
    task_cfg = (cfg.get("eval", {}).get("tasks") or {}).get(args.task) or {}
    task = encode_task(load_task(args.task, **task_cfg), tok)
    if task.kind != "mc":
        raise ValueError(f"{args.task} is not multiple-choice; paired test needs per-question 0/1")

    names = args.configs or [c for c in partial if args.task in partial[c]]
    correct = {}
    for c in names:
        raw = {int(k): v for k, v in partial[c][args.task].items()}
        if len(raw) < len(task.contexts):
            print(f"  skipping {c}: {len(raw)}/{len(task.contexts)} contexts scored")
            continue
        correct[c] = np.array(mc_correct(task, raw)[metric], dtype=np.int8)
    n = len(task.requests)
    print(f"{args.task}: {n} questions, metric={metric}, raw={raw_path}")
    print(f"\n  {'configuration':30s} {metric:>9s}")
    for c, v in correct.items():
        print(f"  {c:30s} {v.mean():9.4f}")

    print(f"\n  {'A':28s} {'B':28s} {'A-B (pts)':>10s} {'95% CI (paired bootstrap)':>26s}"
          f" {'A only':>7s} {'B only':>7s} {'McNemar p':>10s}")
    for a, b in itertools.combinations(correct, 2):
        x, y = correct[a], correct[b]
        d = 100 * (x.mean() - y.mean())
        lo, hi = paired_bootstrap(x, y, args.n_boot, args.seed)
        only_a = int(((x == 1) & (y == 0)).sum())
        only_b = int(((x == 0) & (y == 1)).sum())
        p = mcnemar_exact(only_a, only_b)
        print(f"  {a:28s} {b:28s} {d:+10.2f} {f'[{100*lo:+.2f}, {100*hi:+.2f}]':>26s}"
              f" {only_a:7d} {only_b:7d} {p:10.2g}")


if __name__ == "__main__":
    main()
