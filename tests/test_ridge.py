"""Verify the streaming solve equals a dense reference solve.

The whole memory argument for this project rests on never materializing X. That is
only safe if the streamed Gram gives the same answer as the textbook computation.
"""

import torch

from kvtransfer.gram import GramAccumulator
from kvtransfer.ridge import r2_oos, solve, solve_multi


def _dense_ridge(X, Y, lam):
    """Reference: center, penalize only the slope, solve directly."""
    n = X.shape[0]
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    cxx = Xc.T @ Xc
    scale = float(torch.diagonal(cxx).mean() / n)
    A = cxx + lam * scale * n * torch.eye(X.shape[1], dtype=X.dtype)
    W = torch.linalg.solve(A, Xc.T @ Yc)
    return W, my - mx @ W


def test_streaming_gram_matches_dense():
    torch.manual_seed(0)
    n, d_in, d_out = 5000, 64, 32
    X = torch.randn(n, d_in, dtype=torch.float64)
    Y = X @ torch.randn(d_in, d_out, dtype=torch.float64) + 0.3 + 0.05 * torch.randn(n, d_out, dtype=torch.float64)

    acc = GramAccumulator(d_in, d_out)
    for i in range(0, n, 512):            # streamed in batches
        acc.update(X[i : i + 512], Y[i : i + 512])

    sol = solve(acc, lam=0.01)
    W_ref, b_ref = _dense_ridge(X, Y, 0.01)

    assert torch.allclose(sol.W.double(), W_ref, atol=1e-6, rtol=1e-4)
    assert torch.allclose(sol.b.double(), b_ref, atol=1e-6, rtol=1e-4)


def test_accumulator_merge_is_associative():
    """Shards built separately must sum to the single-pass result."""
    torch.manual_seed(1)
    X = torch.randn(1200, 16, dtype=torch.float64)
    Y = torch.randn(1200, 8, dtype=torch.float64)

    whole = GramAccumulator(16, 8)
    whole.update(X, Y)

    a, b = GramAccumulator(16, 8), GramAccumulator(16, 8)
    a.update(X[:700], Y[:700])
    b.update(X[700:], Y[700:])
    merged = a + b

    assert merged.n == whole.n
    assert torch.allclose(merged.xtx, whole.xtx, atol=1e-9)
    assert torch.allclose(merged.xty, whole.xty, atol=1e-9)


def test_r2_is_one_for_exact_linear_data():
    """Noiseless linear data with tiny lambda must be recovered essentially exactly."""
    torch.manual_seed(2)
    X = torch.randn(4000, 32, dtype=torch.float64)
    W_true = torch.randn(32, 16, dtype=torch.float64)
    Y = X @ W_true + 1.5

    acc = GramAccumulator(32, 16)
    acc.update(X, Y)
    sol = solve(acc, lam=1e-9)

    assert sol.r2 > 0.999999
    assert torch.allclose(sol.W.double(), W_true, atol=1e-4)


def test_identity_mapping_recovered():
    """The degenerate case the end-to-end guardrail relies on: Y == X gives W == I."""
    torch.manual_seed(3)
    X = torch.randn(3000, 48, dtype=torch.float64)
    acc = GramAccumulator(48, 48)
    acc.update(X, X)
    sol = solve(acc, lam=1e-9)

    assert torch.allclose(sol.W.double(), torch.eye(48, dtype=torch.float64), atol=1e-4)
    assert sol.b.abs().max() < 1e-4
    assert sol.r2 > 0.999999


def test_r2_zero_for_unpredictable_target():
    """Independent X and Y must land near R^2 = 0, not negative-huge or spuriously high."""
    torch.manual_seed(4)
    X = torch.randn(8000, 16, dtype=torch.float64)
    Y = torch.randn(8000, 8, dtype=torch.float64)
    acc = GramAccumulator(16, 8)
    acc.update(X, Y)
    sol = solve(acc, lam=0.01)
    assert -0.05 < sol.r2 < 0.05


def test_solve_multi_shares_one_gram():
    torch.manual_seed(5)
    X = torch.randn(2000, 24, dtype=torch.float64)
    Y = X @ torch.randn(24, 12, dtype=torch.float64)
    acc = GramAccumulator(24, 12)
    acc.update(X, Y)

    sols = solve_multi(acc, [1e-3, 1e-2, 1e-1])
    assert set(sols) == {1e-3, 1e-2, 1e-1}
    # Heavier regularization shrinks the weights.
    assert sols[1e-3].W.norm() > sols[1e-1].W.norm()


def test_oos_r2_penalizes_overfit():
    """Held-out R^2 must be able to fall below in-sample R^2 on noisy data."""
    torch.manual_seed(6)
    d = 200
    X = torch.randn(300, d, dtype=torch.float64)     # n barely above d
    Y = torch.randn(300, 4, dtype=torch.float64)     # pure noise target
    acc = GramAccumulator(d, 4)
    acc.update(X, Y)
    sol = solve(acc, lam=1e-6)

    Xh = torch.randn(500, d, dtype=torch.float64)
    Yh = torch.randn(500, 4, dtype=torch.float64)
    assert float(r2_oos(sol, Xh, Yh).mean()) < float(sol.r2_per_dim.mean())


def test_underdetermined_raises():
    acc = GramAccumulator(64, 8)
    acc.update(torch.randn(10, 64, dtype=torch.float64), torch.randn(10, 8, dtype=torch.float64))
    try:
        solve(acc)
    except ValueError as e:
        assert "underdetermined" in str(e)
    else:
        raise AssertionError("expected ValueError for n <= d_in")
