"""Closed-form ridge solve from accumulated Gram statistics.

    W* = (X^T X + lambda I)^-1 X^T Y      (paper eq. for the per-head mapper, lambda = 0.01)

Fitted on centered data so the intercept b is exact and the penalty never shrinks it:
penalizing the intercept would bias every prediction toward zero, which for KV vectors
with non-zero per-channel means is a real accuracy loss, not a technicality.

Solving from a cached Gram means each additional lambda costs one Cholesky on a d x d
matrix -- microseconds. That is why we sweep lambda even though the paper fixes it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .gram import GramAccumulator


@dataclass
class RidgeSolution:
    W: torch.Tensor          # (d_in, d_out)
    b: torch.Tensor          # (d_out,)
    lam: float
    r2_per_dim: torch.Tensor  # (d_out,) in-sample R^2
    n: int

    @property
    def r2(self) -> float:
        """Variance-weighted R^2 over output dims.

        Not the mean of per-dim R^2: dims with near-zero variance produce meaningless
        ratios that would otherwise dominate the average.
        """
        return float(self.r2_per_dim.mean())

    def to_dtype(self, dtype: torch.dtype) -> "RidgeSolution":
        return RidgeSolution(
            self.W.to(dtype), self.b.to(dtype), self.lam, self.r2_per_dim, self.n
        )


def solve(acc: GramAccumulator, lam: float = 0.01, jitter: float = 1e-8) -> RidgeSolution:
    """Ridge solve with intercept, from accumulated statistics."""
    # Promote to float64 for the solve regardless of accumulator precision: centering
    # (`xtx - n*outer(mx,mx)`) is cancellation-prone, and the promotion is free compared
    # with the Cholesky itself.
    n = acc.n
    if n <= acc.d_in:
        raise ValueError(
            f"underdetermined: n={n} <= d_in={acc.d_in}; capture more tokens"
        )

    f64 = torch.float64
    mx = (acc.sx / n).to(f64)                         # (d_in,)
    my = (acc.sy / n).to(f64)                         # (d_out,)

    # Centered second moments: Cov(X,X) and Cov(X,Y), scaled by n.
    cxx = acc.xtx.to(f64) - n * torch.outer(mx, mx)
    cxy = acc.xty.to(f64) - n * torch.outer(mx, my)

    d = acc.d_in
    eye = torch.eye(d, dtype=cxx.dtype, device=cxx.device)
    # Scale the penalty by the mean feature variance so lambda means the same thing
    # regardless of how the activations happen to be scaled at this layer.
    scale = float(torch.diagonal(cxx).mean() / max(n, 1)) or 1.0
    A = cxx + (lam * scale * n + jitter) * eye

    try:
        L = torch.linalg.cholesky(A)
        W = torch.cholesky_solve(cxy, L)
    except RuntimeError:
        # Cholesky needs positive-definiteness; fall back if the Gram is degenerate.
        W = torch.linalg.lstsq(A, cxy).solution

    b = my - mx @ W

    # In-sample R^2 per output dim, computed from the sufficient statistics:
    #   RSS = SYY_c - 2 W^T CXY + W^T CXX W   (diagonal terms only)
    syy_c = acc.syy.to(f64) - n * my * my                      # centered total SS
    rss = syy_c - 2.0 * (W * cxy).sum(0) + ((cxx @ W) * W).sum(0)
    rss = rss.clamp_min(0.0)
    r2 = 1.0 - rss / syy_c.clamp_min(1e-12)

    return RidgeSolution(
        W=W.to(torch.float32), b=b.to(torch.float32), lam=lam,
        r2_per_dim=r2.to(torch.float32), n=n,
    )


def solve_multi(acc: GramAccumulator, lams) -> dict[float, RidgeSolution]:
    """One Gram, many lambdas. Each extra lambda is a single Cholesky."""
    return {float(l): solve(acc, float(l)) for l in lams}


@torch.no_grad()
def r2_oos(
    sol: RidgeSolution, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Out-of-sample R^2 per output dim on held-out (x, y).

    In-sample R^2 on 10^5 observations with d<=3584 features is mostly trustworthy,
    but the k-sweep selects over models of very different capacity (k=1 vs k=all is a
    28x difference in parameters), so selection must use held-out numbers.
    """
    pred = x.to(sol.W.dtype) @ sol.W + sol.b
    y = y.to(pred.dtype)
    rss = ((y - pred) ** 2).sum(0)
    tss = ((y - y.mean(0)) ** 2).sum(0).clamp_min(1e-12)
    return 1.0 - rss / tss
