"""Where the mapped model loses HellaSwag questions the target gets right on its own.

Reads the per-question scores 07_eval_tasks.py saves (eval_tasks.partial.json), splits
questions by standalone vs mapper correctness, and reports what separates the lost ones:
how close standalone's call was, context length, and overlap with zero-copy's errors.
Prints a few lost questions with every configuration's scores for the four endings.

No model is loaded; only the target tokenizer, to rebuild the task's request structure.

    python scripts/12_failure_examples.py --config configs/qwen3_1.7b_to_4b.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np

from kvtransfer.eval.harness import encode_task, task_requests_by_context
from kvtransfer.eval.tasks import load_task
from kvtransfer.pipeline import load_config
from kvtransfer.runtime import output_dir

ROOT = Path(__file__).resolve().parents[1]


def norm_scores(task, raw: dict) -> np.ndarray:
    """(n_docs, n_choices) log-likelihood / continuation length, as acc_norm ranks them."""
    out = np.full((len(task.requests), max(len(r) for r in task.requests)), -np.inf)
    for ci, pairs in task_requests_by_context(task).items():
        for (d, c), s in zip(pairs, raw[ci]):
            out[d, c] = s / len(task.requests[d][c][2])
    return out


def margin(s: np.ndarray) -> np.ndarray:
    """Gap between the top two choices: how decisive the pick was."""
    top2 = np.sort(s, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mapper", default="mapper_k8_lam0.01_allheads")
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None, help="default: <output_dir>/failure_analysis.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = output_dir(cfg, ROOT)
    partial = json.loads((out_dir / "eval_tasks.partial.json").read_text())

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["target"])
    task_obj = load_task("hellaswag")
    task = encode_task(task_obj, tok)
    labels = np.array(task.labels)
    names = ["standalone", args.mapper, "zero_copy"]
    S = {n: norm_scores(task, {int(k): v for k, v in partial[n]["hellaswag"].items()}) for n in names}
    right = {n: S[n].argmax(1) == labels for n in names}
    sa, mp, zc = right["standalone"], right[args.mapper], right["zero_copy"]

    lost, gained = sa & ~mp, ~sa & mp
    n = len(labels)
    print(f"{cfg['source']} -> {cfg['target']}, HellaSwag {n} questions, {args.mapper}")
    print(f"  both right {int((sa & mp).sum())}, both wrong {int((~sa & ~mp).sum())}, "
          f"lost {int(lost.sum())}, gained {int(gained.sum())}, net {100 * (mp.mean() - sa.mean()):+.2f} pts")

    # 1. Near-ties: was standalone's call on the lost questions already close?
    m = margin(S["standalone"])
    med_lost, med_kept = float(np.median(m[lost])), float(np.median(m[sa & mp]))
    print(f"  standalone margin (top-2 gap, nats/char): lost median {med_lost:.4f}, kept median {med_kept:.4f}")
    q = np.quantile(m[sa], [0.25, 0.5, 0.75])
    bins = np.digitize(m, q)
    by_margin = [float(lost[sa & (bins == b)].sum() / max(1, (sa & (bins == b)).sum())) for b in range(4)]
    print("  loss rate by standalone-margin quartile (low -> high): "
          + ", ".join(f"{100 * r:.1f}%" for r in by_margin))

    # 2. Context length: do losses grow with the transferred prefix?
    ctx_len = np.array([len(task.contexts[task.requests[d][0][0]]) for d in range(n)])
    lq = np.quantile(ctx_len[sa], [0.25, 0.5, 0.75])
    lbins = np.digitize(ctx_len, lq)
    by_len = [float(lost[sa & (lbins == b)].sum() / max(1, (sa & (lbins == b)).sum())) for b in range(4)]
    print(f"  context length quartile edges (tokens): {lq.astype(int).tolist()}")
    print("  loss rate by context-length quartile (short -> long): "
          + ", ".join(f"{100 * r:.1f}%" for r in by_len))

    # 3. Overlap with zero-copy: are the lost questions ones the unmapped cache also fails?
    zc_wrong_lost = float((~zc[lost]).mean())
    zc_wrong_kept = float((~zc[sa & mp]).mean())
    print(f"  zero-copy also wrong: on lost {100 * zc_wrong_lost:.1f}%, on kept {100 * zc_wrong_kept:.1f}%")

    # 4. Examples: lost questions where standalone was confident, mapper most confidently wrong.
    conf_wrong = margin(S[args.mapper])
    order = [d for d in np.argsort(-(m * conf_wrong)) if lost[d]][: args.examples]
    examples = []
    for d in order:
        doc = task_obj.docs[d]
        ex = {"doc": int(d), "context": doc.contexts[0], "endings": [c.strip() for c in doc.conts],
              "label": int(labels[d]),
              "scores": {nm: [round(float(x), 4) for x in S[nm][d]] for nm in names},
              "picked": {nm: int(S[nm][d].argmax()) for nm in names}}
        examples.append(ex)
        print(f"\n  doc {d}: {ex['context']}")
        for c, e in enumerate(ex["endings"]):
            tag = "*" if c == ex["label"] else " "
            sc = "  ".join(f"{S[nm][d, c]:+.3f}" for nm in names)
            print(f"   {tag}{c} [{sc}] {e[:110]}")
        print(f"   picked: standalone {ex['picked']['standalone']}, mapper {ex['picked'][args.mapper]}, "
              f"zero-copy {ex['picked']['zero_copy']}  (scores: standalone, mapper, zero-copy)")

    out = args.out or out_dir / "failure_analysis.json"
    out.write_text(json.dumps({
        "source": cfg["source"], "target": cfg["target"], "mapper": args.mapper, "n": n,
        "counts": {"both_right": int((sa & mp).sum()), "both_wrong": int((~sa & ~mp).sum()),
                   "lost": int(lost.sum()), "gained": int(gained.sum())},
        "margin_median": {"lost": med_lost, "kept": med_kept},
        "loss_rate_by_margin_quartile": by_margin,
        "context_len_quartile_edges": lq.astype(int).tolist(),
        "loss_rate_by_context_quartile": by_len,
        "zero_copy_wrong": {"lost": zc_wrong_lost, "kept": zc_wrong_kept},
        "examples": examples,
    }, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
