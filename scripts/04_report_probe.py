"""Turn a probe .pt into the tables the writeup needs.

Prints the head-averaged R^2 heatmap as text, the best source layer per target layer,
and the headline comparisons against the paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from kvtransfer.select import aligned_cells, identity_alignment, rank_sources

ROOT = Path(__file__).resolve().parents[1]

PAPER = {
    "best_single_source_k_stripped": 0.81,   # Qwen3 14B->32B
    "ridge_k": 0.79,
    "ridge_v": 0.65,
}


def heatmap(m: torch.Tensor, title: str, width: int = 28) -> str:
    """Compact text heatmap; rows are source layers, columns target layers."""
    ramp = " .:-=+*#%@"
    lo, hi = float(m.min()), float(m.max())
    rng = max(hi - lo, 1e-9)
    lines = [f"{title}  (min {lo:.2f}  max {hi:.2f})", "      tgt " + "".join(
        f"{t%10}" for t in range(m.shape[1]))]
    for s in range(m.shape[0]):
        row = "".join(ramp[min(int((float(v) - lo) / rng * (len(ramp) - 1)), len(ramp) - 1)]
                      for v in m[s])
        lines.append(f"  src {s:2d} {row}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", nargs="?", default=str(ROOT / "results/probe_r2_n50.pt"))
    args = ap.parse_args()

    d = torch.load(args.probe)
    r2, Ls, Lt = d["r2"], d["Ls"], d["Lt"]
    print(f"{d['source']} -> {d['target']}")
    print(f"{d['n_seqs']} sequences, {d['n_obs']} observations per head\n")

    for kind in ("k_content", "v"):
        m = r2[kind].mean(-1)
        print(heatmap(m, f"=== {kind} head-averaged R^2 ==="))
        print()

    print("=== headline numbers ===")
    for kind in ("k_content", "k_rope", "v"):
        m = r2[kind].mean(-1)
        finite = m[~torch.isnan(m)]
        if finite.numel() == 0:
            continue
        best = float(finite.max())
        diag = aligned_cells(m)
        diag = diag[~torch.isnan(diag)]
        print(f"  {kind:10s}: best {best:.3f}   layer-aligned mean {float(diag.mean()):.3f}")

    kc = aligned_cells(r2["k_content"].mean(-1))
    kr = aligned_cells(r2["k_rope"].mean(-1))
    mask = ~(torch.isnan(kc) | torch.isnan(kr))
    if mask.any():
        print(f"\n  RoPE stripping: K_content {float(kc[mask].mean()):.3f} "
              f"vs K_rope {float(kr[mask].mean()):.3f} "
              f"(positive difference means stripping helps)")

    print("\n=== best source layer per target layer (k_content) ===")
    order = rank_sources(r2["k_content"])
    havg = r2["k_content"].mean(-1)
    align = identity_alignment(Ls, Lt)
    aligned = 0
    for lt in range(Lt):
        b = int(order[lt, 0])
        a = align[lt][0]
        # Within one layer of the proportional alignment counts as aligned when depths
        # differ, since rounding makes the "aligned" source layer ambiguous.
        ok = b == a if Ls == Lt else abs(b - a) <= 1
        aligned += ok
        if lt % 4 == 0 or not ok:
            print(f"  tgt {lt:2d} <- src {b:2d}  (R^2 {float(havg[b, lt]):.3f})"
                  + ("" if ok else f"   [aligned would be src {a}]"))
    print(f"  layer-aligned in {aligned}/{Lt} target layers")

    print("\n=== vs paper (Qwen3 14B->32B) ===")
    ours = float(r2["k_content"].mean(-1).max())
    print(f"  best single-source K R^2:  ours {ours:.3f}  paper {PAPER['best_single_source_k_stripped']:.2f}")
    if ours < 0.4:
        print("\n  Well below the paper. At this scale the two models may simply not")
        print("  share enough linear structure -- a real finding, since the paper's")
        print("  smallest pair is 3B and nothing below that has been reported.")
    elif ours < 0.7:
        print("\n  Partial linear structure: weaker than the paper's 3B+ pairs but")
        print("  clearly non-trivial. Worth fitting top-k mappers to see how much")
        print("  multi-source selection recovers.")
    else:
        print("\n  Comparable to the paper. The method appears to hold at sub-2B scale.")


if __name__ == "__main__":
    main()
