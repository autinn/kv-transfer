"""Out-of-sample R^2 from streamed statistics (03_fit.py --holdout-shards).

The fit's R^2 is in-sample; held-out R^2 is computed from a second set of Gram statistics
over held-out sequences, so it must equal R^2 computed directly on the raw held-out data.
"""

import torch

from kvtransfer.gram import GramAccumulator
from kvtransfer.ridge import r2_from_stats, r2_oos, solve
from kvtransfer.stream import TopKGrams, _per_head


def _data(n, d_in, d_out, seed, shift=0.0):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(d_in, d_out, generator=g)
    x = torch.randn(n, d_in, generator=g) * 3 + shift          # non-zero means on purpose
    y = x @ W + 0.8 * torch.randn(n, d_out, generator=g) + 2.0
    return x, y


def test_r2_from_stats_matches_r2_oos_on_raw_data():
    x_tr, y_tr = _data(400, 12, 5, seed=0)
    x_ho, y_ho = _data(150, 12, 5, seed=0, shift=0.5)            # same map, shifted inputs
    acc = GramAccumulator(12, 5)
    acc.update(x_tr, y_tr)
    sol = solve(acc, lam=0.01)

    ho = GramAccumulator(12, 5)
    for chunk in torch.split(torch.arange(150), 40):              # streamed, uneven batches
        ho.update(x_ho[chunk], y_ho[chunk])
    torch.testing.assert_close(r2_from_stats(ho, sol.W, sol.b), r2_oos(sol, x_ho, y_ho),
                               rtol=1e-5, atol=1e-5)


def test_heldout_r2_is_below_in_sample_when_overfitting():
    """Many features, few observations: in-sample R^2 is inflated, held-out is not."""
    g = torch.Generator().manual_seed(1)
    x_tr, x_ho = torch.randn(60, 40, generator=g), torch.randn(200, 40, generator=g)
    w = torch.zeros(40, 1)
    w[0] = 1.0
    y_tr = x_tr @ w + torch.randn(60, 1, generator=g)
    y_ho = x_ho @ w + torch.randn(200, 1, generator=g)
    acc, ho = GramAccumulator(40, 1), GramAccumulator(40, 1)
    acc.update(x_tr, y_tr)
    ho.update(x_ho, y_ho)
    sol = solve(acc, lam=1e-4)
    assert float(r2_from_stats(ho, sol.W, sol.b)) < sol.r2 - 0.1


def _batches(Ls, Lt, H, D, n_batches, seed):
    g = torch.Generator().manual_seed(seed)
    mix = torch.randn(Ls * H * D, Lt * H * D, generator=g) / (Ls * H * D) ** 0.5
    out = []
    for _ in range(n_batches):
        B, S = 2, 30
        sk, sv = torch.randn(Ls, B, H, S, D, generator=g), torch.randn(Ls, B, H, S, D, generator=g)
        def tgt(src):
            flat = src.permute(1, 3, 0, 2, 4).reshape(B, S, Ls * H * D) @ mix
            flat = flat + 0.3 * torch.randn(flat.shape, generator=g)
            return flat.reshape(B, S, Lt, H, D).permute(2, 0, 3, 1, 4).contiguous()
        out.append((sk, sv, tgt(sk), tgt(sv)))
    return out


def test_topk_heldout_r2_matches_direct_computation_both_modes():
    Ls, Lt, H, D = 3, 2, 2, 4
    src_layers = {0: [0, 2], 1: [1, 2]}
    train, held = _batches(Ls, Lt, H, D, 4, seed=2), _batches(Ls, Lt, H, D, 2, seed=3)
    for all_heads in (False, True):
        g = TopKGrams(src_layers, H, D, all_heads=all_heads)
        ho = TopKGrams(src_layers, H, D, all_heads=all_heads)
        for b in train:
            g.update(*b)
        for b in held:
            ho.update(*b)
        for kind in ("K", "V"):
            for lt, srcs in src_layers.items():
                sols = g.solve_heads(kind, lt, 0.01)
                got = ho.heldout_r2_heads(kind, lt, sols)
                # Direct: build the held-out features and targets by hand, score with r2_oos.
                i = 0 if kind == "K" else 1
                s = torch.cat([_per_head(b[i]) for b in held], dim=2)       # (Ls, H, N, D)
                t = torch.cat([_per_head(b[2 + i]) for b in held], dim=2)   # (Lt, H, N, D)
                for h in range(H):
                    heads = range(H) if all_heads else [h]
                    x = torch.cat([s[ls, h2] for ls in srcs for h2 in heads], dim=-1)
                    torch.testing.assert_close(got[h], r2_oos(sols[h], x, t[lt, h]),
                                               rtol=1e-4, atol=1e-4)
