"""Cross-layer source selection (the paper's top-k step).

For each target layer, the paper picks the k most predictive source layers by
head-averaged R^2 and concatenates them into the feature vector:

    X^l = [K_s^(l1) || K_s^(l2) || ... || K_s^(lk)]

The selection comes from Stage A's single-source probe, so choosing k afterward costs
nothing beyond re-running the Stage B fit at that k.
"""

from __future__ import annotations

import torch


def rank_sources(r2: torch.Tensor) -> torch.Tensor:
    """Rank source layers per target layer by head-averaged R^2.

    r2: (n_src, n_tgt, n_heads). Returns (n_tgt, n_src) with source-layer indices
    ordered best-first for each target layer.
    """
    head_avg = r2.mean(-1)                      # (n_src, n_tgt)
    return head_avg.T.argsort(dim=-1, descending=True)


def top_k(r2: torch.Tensor, k: int) -> dict[int, list[int]]:
    """Best k source layers for each target layer.

    Sources are returned in ascending layer order rather than R^2 order: the ridge fit
    is invariant to feature ordering, and a stable ordering makes the saved mappers
    comparable across k.
    """
    order = rank_sources(r2)
    n_tgt, n_src = order.shape
    k = min(k, n_src)
    return {lt: sorted(order[lt, :k].tolist()) for lt in range(n_tgt)}


def identity_alignment(n_src: int, n_tgt: int) -> dict[int, list[int]]:
    """Baseline: target layer l reads source layer round(l * (n_src-1)/(n_tgt-1)).

    For equal-depth pairs this is exactly layer l -> layer l. Worth fitting alongside
    top-k because it is the obvious prior, and if top-k cannot beat it then cross-layer
    selection is not earning its complexity on this pair.
    """
    if n_tgt == 1:
        return {0: [0]}
    return {
        lt: [min(n_src - 1, round(lt * (n_src - 1) / (n_tgt - 1)))]
        for lt in range(n_tgt)
    }


def selection_summary(r2: torch.Tensor, ks=(1, 2, 4, 8)) -> str:
    """Human-readable report of which source layers win, for the results writeup."""
    order = rank_sources(r2)
    head_avg = r2.mean(-1)
    lines = ["target -> best sources (head-averaged R^2)"]
    for lt in range(order.shape[0]):
        best = order[lt, :4].tolist()
        scores = [f"{head_avg[ls, lt]:.2f}" for ls in best]
        lines.append(
            f"  tgt {lt:2d}: " + ", ".join(f"src {s}({sc})" for s, sc in zip(best, scores))
        )
    return "\n".join(lines)


def aligned_cells(m: torch.Tensor) -> torch.Tensor:
    """Values of an (n_src, n_tgt, ...) grid at the layer-aligned cells.

    For equal depth this is `m.diagonal()`. For unequal depth (14B: 40 layers -> 32B:
    64) `diagonal()` would silently pair target layer 50 with nothing and target layer
    10 with the wrong source, so alignment is proportional, as in `identity_alignment`.
    Returns shape (n_tgt, ...).
    """
    n_src, n_tgt = m.shape[0], m.shape[1]
    align = identity_alignment(n_src, n_tgt)
    return torch.stack([m[align[lt][0], lt] for lt in range(n_tgt)])
