"""The batched grid must agree with the per-cell accumulator it replaces.

The batched form exists for memory and speed, not for different math: it collapses
12,768 small accumulators into a handful of large tensors. If it disagreed with the
reference implementation even slightly, every R^2 in the writeup would be suspect, so
this equivalence is checked directly rather than assumed.
"""

import torch

from kvtransfer.batched_gram import BatchedGrid, DiagonalGrid
from kvtransfer.gram import GramAccumulator
from kvtransfer.ridge import solve

Ls, Lt, H, D, N = 3, 4, 2, 8, 500


def _data(seed=0):
    torch.manual_seed(seed)
    # Correlated, offset features, like real activations rather than white noise.
    A = torch.randn(D, D, dtype=torch.float64)
    x = (torch.randn(Ls, H, N, D, dtype=torch.float64) @ A) * 3 + 1.5
    y = (torch.randn(Lt, H, N, D, dtype=torch.float64) @ A) * 2 - 0.7
    # Make some cells genuinely predictable so R^2 spans a useful range.
    y[0] = x[0] @ torch.randn(D, D, dtype=torch.float64) * 0.5 + 0.1
    y[1] = x[min(1, Ls - 1)] * 0.8 + 0.3
    return x, y


def test_batched_matches_per_cell_r2():
    x, y = _data()
    grid = BatchedGrid(Ls, Lt, H, D, dtype=torch.float64)
    grid.update(x, y)
    got = grid.solve_r2(lam=0.01)          # (Ls, Lt, H)

    assert got.shape == (Ls, Lt, H)
    for ls in range(Ls):
        for lt in range(Lt):
            for h in range(H):
                acc = GramAccumulator(D, D)
                acc.update(x[ls, h], y[lt, h])
                ref = solve(acc, lam=0.01).r2
                assert abs(float(got[ls, lt, h]) - ref) < 1e-6, (
                    f"cell ({ls},{lt},{h}): batched {float(got[ls,lt,h]):.8f} vs {ref:.8f}"
                )


def test_batched_accumulates_across_chunks():
    """Streaming in chunks must equal a single-shot update."""
    x, y = _data(1)
    whole = BatchedGrid(Ls, Lt, H, D, dtype=torch.float64)
    whole.update(x, y)

    chunked = BatchedGrid(Ls, Lt, H, D, dtype=torch.float64)
    for i in range(0, N, 125):
        chunked.update(x[:, :, i : i + 125], y[:, :, i : i + 125])

    assert chunked.n == whole.n
    assert torch.allclose(chunked.xtx, whole.xtx, rtol=1e-9, atol=1e-6)
    assert torch.allclose(chunked.xty, whole.xty, rtol=1e-9, atol=1e-6)
    assert torch.allclose(chunked.solve_r2(), whole.solve_r2(), atol=1e-6)


def test_diagonal_grid_matches_batched_diagonal():
    x, y = _data(2)
    L = min(Ls, Lt)
    dg = DiagonalGrid(L, H, D, dtype=torch.float64)
    dg.update(x[:L], y[:L])
    got = dg.solve_r2(lam=0.01)            # (L, H)

    full = BatchedGrid(L, L, H, D, dtype=torch.float64)
    full.update(x[:L], y[:L])
    ref = full.solve_r2(lam=0.01)
    for l in range(L):
        for h in range(H):
            assert abs(float(got[l, h]) - float(ref[l, l, h])) < 1e-6


def test_identity_target_gives_r2_one():
    """Predicting a source layer from itself must be perfect."""
    x, _ = _data(3)
    grid = BatchedGrid(Ls, Ls, H, D, dtype=torch.float64)
    grid.update(x, x)
    r2 = grid.solve_r2(lam=1e-9)
    for l in range(Ls):
        assert float(r2[l, l].min()) > 0.9999, f"layer {l}: {float(r2[l,l].min())}"


def test_float32_grid_close_to_float64():
    """float32 accumulation is the memory-saving default; confirm it costs nothing real."""
    x, y = _data(4)
    g64 = BatchedGrid(Ls, Lt, H, D, dtype=torch.float64)
    g64.update(x, y)
    g32 = BatchedGrid(Ls, Lt, H, D, dtype=torch.float32)
    g32.update(x, y)
    assert torch.allclose(g64.solve_r2(), g32.solve_r2(), atol=1e-4)


def test_allocation_count_is_small():
    """The whole point: a handful of tensors, not one per cell."""
    grid = BatchedGrid(28, 28, 8, 128)
    tensors = [v for v in vars(grid).values() if isinstance(v, torch.Tensor)]
    assert len(tensors) == 5, f"expected 5 big tensors, got {len(tensors)}"
    total = sum(t.numel() * t.element_size() for t in tensors)
    # xty dominates: 28*28*8*128*128 * 4 bytes.
    assert total < 1.6e9, f"{total/1e9:.2f} GB is larger than intended"
