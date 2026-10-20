"""Figures for the write-up and progress report, from results already on disk.

Reads the per-pair probe / held-out / HellaSwag outputs under results/ and writes
PNGs plus one JSON (the numbers behind every figure) to results/plots/writeup/.
No model is loaded; runs in a few seconds on CPU.

    python scripts/09_writeup_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "plots" / "writeup"

# Fixed series -> colour mapping, shared by every figure (categorical slots 1-3).
C_K8, C_K1, C_ZC = "#2a78d6", "#eb6834", "#1baf7a"
C_REF, C_INK, C_MUTED, C_GRID = "#52514e", "#0b0b0b", "#898781", "#e1e0d9"

# 0.6B -> 1.7B all-heads numbers come from the 2026-10-06 Modal runs (devlog); their
# raw outputs were not copied into results/. Same-head numbers are REPORT3's.
SMALL_PAIR = {
    "hellaswag": {"standalone": 0.6031, "oracle": 0.6031, "zero_copy": 0.3381,
                  "k1": 0.5384, "k8": 0.5563, "k1_samehead": 0.4899, "k8_samehead": 0.5369},
    "r2": {"k1": {"train_k": 0.656, "held_k": 0.643, "train_v": 0.550, "held_v": 0.534},
           "k8": {"train_k": 0.744, "held_k": 0.710, "train_v": 0.637, "held_v": 0.592}},
}

PAIRS = [
    {"key": "0.6b_1.7b", "label": "0.6B → 1.7B", "change": "width", "Ls": 28, "Lt": 28,
     "probe": None, "eval": None, "heldout": None},
    {"key": "1.7b_4b", "label": "1.7B → 4B", "change": "depth + width", "Ls": 28, "Lt": 36,
     "probe": "qwen3_1.7b_to_4b/probe_r2_n500.pt", "eval": "qwen3_1.7b_to_4b/eval_tasks.json",
     "heldout": "qwen3_1.7b_to_4b/heldout_r2.json"},
    {"key": "4b_8b", "label": "4B → 8B", "change": "width", "Ls": 36, "Lt": 36,
     "probe": "qwen3_4b_to_8b/probe_r2_n500.pt", "eval": "qwen3_4b_to_8b/eval_tasks.json",
     "heldout": "qwen3_4b_to_8b/heldout_r2.json"},
    {"key": "14b_32b", "label": "14B → 32B", "change": "depth", "Ls": 40, "Lt": 64,
     "probe": "qwen3_14b_to_32b/probe_r2_n500.pt", "eval": "qwen3_14b_to_32b/eval_tasks.json",
     "heldout": "qwen3_14b_to_32b_allheads_a100/heldout_r2.json",
     "eval_allheads": "qwen3_14b_to_32b_allheads_a100/eval_tasks.json"},
]


def acc(d: dict, cfg: str):
    return d[cfg]["hellaswag"]["acc_norm"] if cfg in d else None


def load_pair(p: dict) -> dict:
    """Collect HellaSwag acc_norm, held-out R² and the probe grid for one pair."""
    out = {"label": p["label"], "change": p["change"], "Ls": p["Ls"], "Lt": p["Lt"]}
    if p["eval"] is None:
        out["hellaswag"] = dict(SMALL_PAIR["hellaswag"])
        out["r2"] = SMALL_PAIR["r2"]
    else:
        d = json.loads((RES / p["eval"]).read_text())
        h = {"standalone": acc(d, "standalone"), "oracle": acc(d, "oracle"),
             "zero_copy": acc(d, "zero_copy"), "mean_pred": acc(d, "mean_pred"),
             "k1": acc(d, "mapper_k1_lam0.01_allheads"), "k8": acc(d, "mapper_k8_lam0.01_allheads")}
        if "eval_allheads" in p:  # 14B -> 32B: mappers re-scored on one GPU (fe98209)
            a = json.loads((RES / p["eval_allheads"]).read_text())
            h["k8"] = acc(a, "mapper_k8_lam0.01_allheads")
            h["k8_samehead"] = acc(a, "mapper_k8_lam0.01")
        out["hellaswag"] = h
        r2 = {}
        for name, v in json.loads((RES / p["heldout"]).read_text()).items():
            if v["all_heads"]:
                r2[v["k"]] = {"train_k": v["train_r2_k"], "held_k": v["heldout_r2_k"],
                              "train_v": v["train_r2_v"], "held_v": v["heldout_r2_v"]}
        out["r2"] = r2
    h = out["hellaswag"]
    out["retention"] = {c: (100 * h[c] / h["standalone"] if h.get(c) is not None else None)
                        for c in h}
    if p["probe"] is not None:
        pr = torch.load(RES / p["probe"], weights_only=False)
        grid = pr["r2"]["k_content"].float().mean(-1).numpy()  # (Ls, Lt), head-averaged
        out["probe_k"] = grid.round(4).tolist()
        out["best_src"] = grid.argmax(0).tolist()
        out["best_src_r2"] = grid.max(0).round(4).tolist()
    else:
        # Probe .pt not on this machine; REPORT3 records the best source as the
        # layer-aligned one for all 28 target layers.
        out["probe_k"] = None
        out["best_src"] = list(range(p["Lt"]))
        out["best_src_r2"] = None
    return out


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_MUTED)
    ax.tick_params(colors=C_REF, labelsize=9)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


def fig_retention(data):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    style(ax)
    series = [("zero_copy", "Zero-copy (unmapped)", C_ZC), ("k1", "Mapped, k = 1", C_K1),
              ("k8", "Mapped, k = 8", C_K8)]
    w, x = 0.26, np.arange(len(data))
    for i, (cfg, name, col) in enumerate(series):
        vals = [d["retention"].get(cfg) for d in data]
        for j, v in enumerate(vals):
            xx = x[j] + (i - 1) * w
            if v is None:
                ax.text(xx, 3, "not\nrun", ha="center", va="bottom", fontsize=7, color=C_MUTED)
                continue
            ax.bar(xx, v, w - 0.03, color=col, label=name if j == 0 else None, zorder=2)
            if cfg == "k8":
                ax.text(xx, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5,
                        color=C_INK, fontweight="bold")
    ax.axhline(100, color=C_REF, linewidth=1, linestyle="--", zorder=1)
    ax.text(-0.45, 101, "standalone target = 100%", ha="left", va="bottom",
            fontsize=8, color=C_REF)
    ax.set_xticks(x, [f"{d['label']}\n({d['change']})" for d in data], fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("HellaSwag retention (%)", color=C_REF, fontsize=9)
    ax.set_title("Accuracy kept when the big model reads a transferred cache",
                 loc="left", fontsize=11, color=C_INK)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", ncol=3, bbox_to_anchor=(1, -0.32))
    save(fig, "fig1_retention.png")


def fig_alignment(data):
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.9))
    for ax, d in zip(axes, data):
        style(ax)
        ax.yaxis.grid(False)
        Ls, Lt = d["Ls"], d["Lt"]
        t = np.arange(Lt)
        ax.plot(t, t * (Ls - 1) / (Lt - 1), color=C_MUTED, linestyle="--", linewidth=1.2,
                label="proportional")
        ax.plot(t, d["best_src"], color=C_K8, linewidth=2, marker="o", markersize=2.8,
                label="best source (probe)")
        ax.set_title(f"{d['label']}  ({Ls}→{Lt} layers)", fontsize=9.5, color=C_INK, loc="left")
        ax.set_xlabel("target layer", fontsize=8.5, color=C_REF)
        ax.set_xlim(-1, Lt)
        ax.set_ylim(-1, Ls)
    axes[0].set_ylabel("best source layer", fontsize=8.5, color=C_REF)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("Which small-model layer best predicts each big-model layer",
                 x=0.06, ha="left", fontsize=11, color=C_INK, y=1.04)
    fig.tight_layout()
    save(fig, "fig2_alignment.png")


def fig_heatmaps(data):
    have = [d for d in data if d["probe_k"] is not None]
    fig, axes = plt.subplots(1, len(have), figsize=(3.4 * len(have), 3.3))
    for ax, d in zip(axes, have):
        g = np.array(d["probe_k"])
        im = ax.imshow(g, origin="lower", cmap="Blues", vmin=0, vmax=0.8, aspect="auto")
        ax.set_title(d["label"], fontsize=9.5, color=C_INK, loc="left")
        ax.set_xlabel("target layer", fontsize=8.5, color=C_REF)
        ax.tick_params(labelsize=8, colors=C_REF)
    axes[0].set_ylabel("source layer", fontsize=8.5, color=C_REF)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("key R² (single source layer, head-averaged)", fontsize=8, color=C_REF)
    cb.ax.tick_params(labelsize=8, colors=C_REF)
    fig.suptitle("Linear-structure probe: how well each source layer predicts each target layer",
                 x=0.06, ha="left", fontsize=11, color=C_INK, y=1.03)
    save(fig, "fig3_probe_heatmaps.png")


def fig_heads(data):
    rows = []
    s = data[0]["hellaswag"]
    rows.append(("0.6B → 1.7B, k = 1", s["k1_samehead"], s["k1"], s["standalone"]))
    rows.append(("0.6B → 1.7B, k = 8", s["k8_samehead"], s["k8"], s["standalone"]))
    b = data[3]["hellaswag"]
    rows.append(("14B → 32B, k = 8", b["k8_samehead"], b["k8"], b["standalone"]))
    fig, ax = plt.subplots(figsize=(6.6, 2.4))
    style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.8)
    for i, (name, same, allh, sa) in enumerate(rows):
        a, c = 100 * same / sa, 100 * allh / sa
        ax.plot([a, c], [i, i], color=C_GRID, linewidth=3, zorder=1)
        ax.scatter([a], [i], s=60, color=C_K1, zorder=2, label="same head only" if i == 0 else None)
        ax.scatter([c], [i], s=60, color=C_K8, zorder=2, label="all source heads" if i == 0 else None)
        ax.text(c + 0.4, i, f"{c:.1f}%  (+{c - a:.1f})", va="center", fontsize=8.5, color=C_INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(78, 103)
    ax.set_xlabel("HellaSwag retention (%)", fontsize=9, color=C_REF)
    ax.set_title("Same-head vs all-heads features (the paper's layout)",
                 loc="left", fontsize=10.5, color=C_INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.28))
    save(fig, "fig4_heads.png")


def fig_heldout(data):
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    style(ax)
    x = np.arange(len(data))
    for kk, col, off in (("k1", C_K1, -0.12), ("k8", C_K8, 0.12)):
        for j, d in enumerate(data):
            r = d["r2"].get(kk)
            if r is None:
                ax.text(x[j] + off, 0.60, "k = 1\nnot run", ha="center", fontsize=7, color=C_MUTED)
                continue
            ax.plot([x[j] + off] * 2, [r["held_k"], r["train_k"]], color=col, linewidth=2)
            ax.scatter([x[j] + off], [r["train_k"]], s=36, facecolor="white", edgecolor=col,
                       linewidth=1.6, zorder=3, label=f"{kk.replace('k', 'k = ')} in-sample" if j == 1 else None)
            ax.scatter([x[j] + off], [r["held_k"]], s=36, color=col, zorder=3,
                       label=f"{kk.replace('k', 'k = ')} held-out" if j == 1 else None)
    ax.axhline(0.79, color=C_REF, linestyle="--", linewidth=1)
    ax.text(0.5, 0.795, "paper, 14B → 32B, k = 8: 0.79 (protocol not stated)", fontsize=8, color=C_REF,
            va="bottom", ha="center")
    ax.set_xticks(x, [d["label"] for d in data], fontsize=9)
    ax.set_ylim(0.5, 0.85)
    ax.set_ylabel("key R² (mean over layers, heads)", fontsize=9, color=C_REF)
    ax.set_title("Mapper fit: in-sample vs held-out", loc="left", fontsize=11, color=C_INK)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save(fig, "fig5_heldout_r2.png")


def main():
    data = [load_pair(p) for p in PAIRS]
    fig_retention(data)
    fig_alignment(data)
    fig_heatmaps(data)
    fig_heads(data)
    fig_heldout(data)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "writeup_data.json").write_text(json.dumps(data))
    print("wrote", OUT / "writeup_data.json")


if __name__ == "__main__":
    main()
