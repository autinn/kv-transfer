"""Streaming sufficient statistics for ridge regression.

The design matrix we would need to hold is (N x d) with N ~= 128K observations per head
and d = k * 128. At k=8 that is 0.5 GB per head in fp32, times 28 layers x 8 heads. We
never build it. Instead we accumulate the normal-equation terms in one pass:

    XtX (d x d),  XtY (d x d_out),  plus sums and counts for the intercept

At k=8 the Gram is 4 MB, and it is *shared across all heads of a target layer* because
every head of a given layer regresses on the same source features. Memory stops being
the constraint; this is the whole reason the paper's closed form is tractable on a laptop.

All statistics are float32. Model forwards run in bf16, but accumulating 10^5 outer
products in bf16 would destroy the Gram's conditioning before we ever reach the solve.
"""

from __future__ import annotations

import torch


class GramAccumulator:
    """Accumulates XtX, XtY, sum(X), sum(Y), sum(Y*Y), N over streamed batches.

    Centering is handled at solve time from the accumulated sums rather than by
    pre-centering the data, so a single pass suffices and the fit gets an intercept
    for free.
    """

    def __init__(self, d_in: int, d_out: int, device="cpu", dtype=torch.float64):
        """`dtype` is the accumulator precision, not the input precision.

        float64 is the safe default: centering (`xtx - n*mx*mx`) is a cancellation-prone
        step. float32 halves memory, which matters when thousands of accumulators are
        live at once, and measured at our scale (n~1.3e4, |x|~4e2, correlated features)
        it shifts R^2 by under 1e-6 -- irrelevant for a diagnostic reported to three
        decimals, though float64 remains the right choice for fitting production maps.
        """
        self.d_in, self.d_out = d_in, d_out
        self.dtype = dtype
        f = dict(dtype=dtype, device=device)
        self.xtx = torch.zeros(d_in, d_in, **f)
        self.xty = torch.zeros(d_in, d_out, **f)
        self.sx = torch.zeros(d_in, **f)
        self.sy = torch.zeros(d_out, **f)
        self.syy = torch.zeros(d_out, **f)
        self.n = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """x: (n, d_in), y: (n, d_out). Any dtype; cast to the accumulator dtype."""
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"batch mismatch: x {x.shape[0]} vs y {y.shape[0]}")
        x = x.to(self.dtype)
        y = y.to(self.dtype)
        self.xtx += x.T @ x
        self.xty += x.T @ y
        self.sx += x.sum(0)
        self.sy += y.sum(0)
        self.syy += (y * y).sum(0)
        self.n += x.shape[0]

    def state_dict(self) -> dict:
        return {
            "xtx": self.xtx, "xty": self.xty, "sx": self.sx,
            "sy": self.sy, "syy": self.syy,
            "n": torch.tensor(self.n), "d_in": torch.tensor(self.d_in),
            "d_out": torch.tensor(self.d_out),
        }

    @classmethod
    def from_state_dict(cls, sd: dict) -> "GramAccumulator":
        dt = sd["xtx"].dtype
        acc = cls(int(sd["d_in"]), int(sd["d_out"]), dtype=dt)
        for k in ("xtx", "xty", "sx", "sy", "syy"):
            setattr(acc, k, sd[k].to(dt))
        acc.n = int(sd["n"])
        return acc

    def __add__(self, other: "GramAccumulator") -> "GramAccumulator":
        """Merge shards (accumulation is associative, so shards can be built in parallel)."""
        if (self.d_in, self.d_out) != (other.d_in, other.d_out):
            raise ValueError("cannot merge accumulators of different shape")
        out = GramAccumulator(self.d_in, self.d_out, dtype=self.dtype)
        for k in ("xtx", "xty", "sx", "sy", "syy"):
            setattr(out, k, getattr(self, k) + getattr(other, k))
        out.n = self.n + other.n
        return out
