"""Generate plots from probe, mapper, and perplexity results."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from kvtransfer.mapper import KVMapper
from kvtransfer.select import rank_sources


ROOT = Path(__file__).resolve().parents[1]


def save(fig, out: Path) -> None:
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_probe(probe: dict, outdir: Path) -> None:
    r2 = probe["r2"]
    matrices = [
        ("k_content", "Keys: content-space R^2"),
        ("v", "Values: R^2"),
        ("k_rope", "Keys: post-RoPE R^2"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    image = None
    for ax, (kind, title) in zip(axes, matrices):
        matrix = r2[kind].mean(-1).numpy()
        image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("Target layer")
        ax.set_ylabel("Source layer")
    fig.colorbar(image, ax=axes, label="R^2", shrink=0.82)
    save(fig, outdir / "probe_r2_heatmaps.png")

    k = r2["k_content"].mean(-1)
    order = rank_sources(r2["k_content"])
    best = order[:, 0].numpy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    targets = np.arange(len(best))
    ax.plot(targets, targets, "--", color="0.55", label="layer-aligned")
    ax.plot(targets, best, "o-", color="#1769aa", label="best source layer")
    ax.set(xlabel="Target layer", ylabel="Selected source layer", title="Best source layer per target")
    ax.set_ylim(-0.8, max(len(best) - 1, 1) + 0.8)
    ax.legend()
    save(fig, outdir / "best_source_layer_alignment.png")

    diagonal = k.diagonal().numpy()
    all_values = k.flatten().numpy()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.boxplot([all_values, diagonal], labels=["All layer pairs", "Aligned pairs"], showmeans=True)
    ax.set(ylabel="R^2", title="Distribution of key linear fit quality")
    save(fig, outdir / "probe_r2_distributions.png")

    head_values = r2["k_content"].permute(2, 0, 1).reshape(r2["k_content"].shape[2], -1).numpy()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot([head_values[h] for h in range(head_values.shape[0])], labels=[f"H{h}" for h in range(head_values.shape[0])], showmeans=True)
    ax.set(xlabel="KV head", ylabel="R^2", title="Key R^2 variation across KV heads")
    save(fig, outdir / "probe_r2_by_head.png")


def mapper_info(mapper_dir: Path) -> list[tuple[str, KVMapper, int, float]]:
    entries = []
    for path in sorted(mapper_dir.glob("mapper_*.pt")):
        mapper = KVMapper.from_state_dict(torch.load(path, map_location="cpu"))
        match = re.fullmatch(r"mapper_(.+)_lam([0-9.]+)", path.stem)
        if not match:
            continue
        name, lam_text = match.groups()
        k = len(next(iter(mapper.src_layers.values())))
        entries.append((name, mapper, k, float(lam_text)))
    return entries


def plot_mappers(entries: list[tuple[str, KVMapper, int, float]], outdir: Path) -> None:
    if not entries:
        return
    labels = [name for name, _, _, _ in entries]
    key_r2 = [mapper.mean_r2("K") for _, mapper, _, _ in entries]
    value_r2 = [mapper.mean_r2("V") for _, mapper, _, _ in entries]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.55), 4.8))
    ax.bar(x - width / 2, key_r2, width, label="Keys")
    ax.bar(x + width / 2, value_r2, width, label="Values")
    ax.set_xticks(x, labels, rotation=55, ha="right")
    ax.set(ylabel="Mean in-sample R^2", title="Fitted mapper quality")
    ax.legend()
    save(fig, outdir / "mapper_fit_r2.png")

    grouped = {}
    for name, mapper, k, lam in entries:
        grouped.setdefault(lam, []).append((k, mapper.mean_r2("K"), mapper.mean_r2("V")))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lam, values in sorted(grouped.items()):
        values.sort()
        ax.plot([v[0] for v in values], [v[1] for v in values], "o-", label=f"lambda={lam:g}")
    ax.set(xlabel="Number of source layers (k)", ylabel="Mean key R^2", title="Mapper fit quality vs source-layer count")
    ax.legend()
    save(fig, outdir / "mapper_r2_vs_k.png")

    selected = [item for item in entries if item[2] in {1, 4, 8, 12, 16} and abs(item[3] - 0.01) < 1e-9]
    if selected:
        fig, axes = plt.subplots(len(selected), 1, figsize=(8, max(2.2 * len(selected), 4)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, (name, mapper, k, _) in zip(axes, sorted(selected, key=lambda item: item[2])):
            matrix = np.full((len(mapper.src_layers), k), np.nan)
            for target, sources in mapper.src_layers.items():
                matrix[target, :] = sources
            ax.imshow(matrix, origin="lower", aspect="auto", cmap="tab20", vmin=0, vmax=len(mapper.src_layers) - 1)
            ax.set_ylabel(f"k={k}\ntarget")
            ax.set_yticks([])
        axes[-1].set_xlabel("Selected source rank")
        fig.suptitle("Source layers selected by each top-k mapper")
        save(fig, outdir / "mapper_source_selections.png")


def plot_eval(eval_path: Path, outdir: Path) -> None:
    results = json.loads(eval_path.read_text())
    preferred = ["target_standalone", "oracle_graft", "zero_copy", "mean_pred"]
    controls = [name for name in preferred if name in results]
    mappers = [name for name in results if name.startswith("mapper_")]

    names = controls + sorted(mappers)
    values = [results[name] for name in names]
    colors = ["#2b8cbe" if name in controls[:2] else "#d95f0e" if name == "zero_copy" else "#636363" if name == "mean_pred" else "#31a354" for name in names]
    fig, ax = plt.subplots(figsize=(max(9, len(names) * 0.55), 5))
    ax.bar(np.arange(len(names)), values, color=colors)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(names)), names, rotation=60, ha="right")
    ax.set_ylabel("Perplexity, log scale")
    ax.set_title("Target-cache transfer evaluation")
    save(fig, outdir / "perplexity_comparison.png")

    points = []
    for name in mappers:
        match = re.fullmatch(r"mapper_k(\d+)_lam([0-9.]+)", name)
        if match:
            points.append((int(match.group(1)), float(match.group(2)), results[name]))
    if points:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for lam in sorted({point[1] for point in points}):
            series = sorted(point for point in points if point[1] == lam)
            ax.plot([point[0] for point in series], [point[2] for point in series], "o-", label=f"lambda={lam:g}")
        ax.set_yscale("log")
        ax.set(xlabel="Number of source layers (k)", ylabel="Perplexity, log scale", title="Transfer quality across the mapper sweep")
        ax.legend()
        save(fig, outdir / "perplexity_vs_k_lambda.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, default=ROOT / "results/probe_n100.pt")
    parser.add_argument("--eval", type=Path, default=ROOT / "results/eval_ppl.json")
    parser.add_argument("--mappers", type=Path, default=ROOT / "results/mappers")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/plots")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    plot_probe(torch.load(args.probe, map_location="cpu"), args.outdir)
    plot_mappers(mapper_info(args.mappers), args.outdir)
    plot_eval(args.eval, args.outdir)
    print(f"saved plots to {args.outdir}")


if __name__ == "__main__":
    main()