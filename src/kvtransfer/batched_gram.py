"""Batched Gram accumulation over the whole (src_layer, tgt_layer, head) grid.

The per-cell `GramAccumulator` is correct but allocates one object per cell. For the
full probe grid that is 28*28*8*2 = 12,768 small tensors totalling 1.7 GB, which
fragments the heap badly enough to drive an 8 GB machine into swap even though the
nominal total is modest. It is also slow: 12,768 Python-level calls per chunk.

Here the same statistics live in a handful of large contiguous tensors and are updated
with one `einsum` per chunk:

    xtx[ls, h]     = sum_t  x[ls, h, t] x[ls, h, t]^T        (src layer, head)
    xty[ls, lt, h] = sum_t  x[ls, h, t] y[lt, h, t]^T        (cross term)

Note that `xtx` does not depend on the target layer -- every target regresses on the
same source features -- so it is stored once per (src layer, head) rather than per cell.
That is a 28x saving on the largest array, on top of the allocation-count win.
"""

from __future__ import annotations

import torch


class BatchedGrid:
    """Sufficient statistics for every (src_layer, tgt_layer, head) single-source fit.

    Shapes, with Ls source layers, Lt target layers, H heads and D head dims:
        xtx (Ls, H, D, D)       source second moments, shared across target layers
        xty (Ls, Lt, H, D, D)   cross moments
        sx  (Ls, H, D)          source sums
        sy  (Lt, H, D)          target sums
        syy (Lt, H, D)          target sums of squares
    """

    def __init__(self, Ls: int, Lt: int, H: int, D: int, dtype=torch.float32, device="cpu"):
        self.Ls, self.Lt, self.H, self.D = Ls, Lt, H, D
        self.dtype = dtype
        f = dict(dtype=dtype, device=device)
        self.xtx = torch.zeros(Ls, H, D, D, **f)
        self.xty = torch.zeros(Ls, Lt, H, D, D, **f)
        self.sx = torch.zeros(Ls, H, D, **f)
        self.sy = torch.zeros(Lt, H, D, **f)
        self.syy = torch.zeros(Lt, H, D, **f)
        self.n = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """x: (Ls, H, N, D) source, y: (Lt, H, N, D) target, same N tokens."""
        if x.shape[2] != y.shape[2]:
            raise ValueError(f"token count mismatch: {x.shape[2]} vs {y.shape[2]}")
        x = x.to(self.dtype)
        y = y.to(self.dtype)
        self.xtx += torch.einsum("lhnd,lhne->lhde", x, x)
        self.xty += torch.einsum("shnd,thne->sthde", x, y)
        self.sx += x.sum(2)
        self.sy += y.sum(2)
        self.syy += (y * y).sum(2)
        self.n += x.shape[2]

    @torch.no_grad()
    def solve_r2(self, lam: float = 0.01, jitter: float = 1e-8) -> torch.Tensor:
        """Ridge-fit every cell and return in-sample R^2 of shape (Ls, Lt, H).

        Solved in float64 with a batched Cholesky: the centering step
        (`xtx - n*outer(mx,mx)`) is cancellation-prone, and promoting costs little
        next to the solve itself.
        """
        n = self.n
        if n <= self.D:
            raise ValueError(f"underdetermined: n={n} <= D={self.D}")
        f64 = torch.float64
        Ls, Lt, H, D = self.Ls, self.Lt, self.H, self.D

        mx = (self.sx / n).to(f64)                      # (Ls,H,D)
        my = (self.sy / n).to(f64)                      # (Lt,H,D)

        cxx = self.xtx.to(f64) - n * torch.einsum("lhd,lhe->lhde", mx, mx)
        cxy = self.xty.to(f64) - n * torch.einsum("shd,the->sthde", mx, my)
        syy_c = self.syy.to(f64) - n * my * my          # (Lt,H,D)

        # Penalty scaled by mean feature variance so lambda means the same thing at
        # every layer regardless of activation scale.
        scale = torch.diagonal(cxx, dim1=-2, dim2=-1).mean(-1) / n   # (Ls,H)
        eye = torch.eye(D, dtype=f64)
        A = cxx + (lam * scale.clamp_min(1e-12) * n + jitter)[..., None, None] * eye

        L = torch.linalg.cholesky(A)                    # (Ls,H,D,D)
        # Broadcast the per-source factorization across target layers.
        Lb = L[:, None].expand(Ls, Lt, H, D, D).reshape(-1, D, D)
        W = torch.cholesky_solve(cxy.reshape(-1, D, D), Lb).reshape(Ls, Lt, H, D, D)

        rss = (
            syy_c[None]                                              # (1,Lt,H,D)
            - 2.0 * (W * cxy).sum(-2)
            + (torch.einsum("shde,sthef->sthdf", cxx, W) * W).sum(-2)
        ).clamp_min(0.0)
        r2 = 1.0 - rss / syy_c[None].clamp_min(1e-12)
        return r2.mean(-1).to(torch.float32)            # average over output dims


class DiagonalGrid:
    """Same statistics, but only where source layer == target layer.

    Used for the post-RoPE key comparison, whose purpose is the single question
    "does stripping RoPE improve the linear fit?" -- answered on the layer-aligned
    diagonal at 1/Ls the cost of the full grid.
    """

    def __init__(self, L: int, H: int, D: int, dtype=torch.float32, device="cpu"):
        self.L, self.H, self.D = L, H, D
        self.dtype = dtype
        f = dict(dtype=dtype, device=device)
        self.xtx = torch.zeros(L, H, D, D, **f)
        self.xty = torch.zeros(L, H, D, D, **f)
        self.sx = torch.zeros(L, H, D, **f)
        self.sy = torch.zeros(L, H, D, **f)
        self.syy = torch.zeros(L, H, D, **f)
        self.n = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """x, y: (L, H, N, D), matched layer for layer."""
        x = x.to(self.dtype)
        y = y.to(self.dtype)
        self.xtx += torch.einsum("lhnd,lhne->lhde", x, x)
        self.xty += torch.einsum("lhnd,lhne->lhde", x, y)
        self.sx += x.sum(2)
        self.sy += y.sum(2)
        self.syy += (y * y).sum(2)
        self.n += x.shape[2]

    @torch.no_grad()
    def solve_r2(self, lam: float = 0.01, jitter: float = 1e-8) -> torch.Tensor:
        """Returns R^2 of shape (L, H)."""
        n = self.n
        f64 = torch.float64
        D = self.D
        mx = (self.sx / n).to(f64)
        my = (self.sy / n).to(f64)
        cxx = self.xtx.to(f64) - n * torch.einsum("lhd,lhe->lhde", mx, mx)
        cxy = self.xty.to(f64) - n * torch.einsum("lhd,lhe->lhde", mx, my)
        syy_c = self.syy.to(f64) - n * my * my

        scale = torch.diagonal(cxx, dim1=-2, dim2=-1).mean(-1) / n
        eye = torch.eye(D, dtype=f64)
        A = cxx + (lam * scale.clamp_min(1e-12) * n + jitter)[..., None, None] * eye
        W = torch.cholesky_solve(cxy, torch.linalg.cholesky(A))

        rss = (
            syy_c - 2.0 * (W * cxy).sum(-2)
            + (torch.einsum("lhde,lhef->lhdf", cxx, W) * W).sum(-2)
        ).clamp_min(0.0)
        return (1.0 - rss / syy_c.clamp_min(1e-12)).mean(-1).to(torch.float32)
