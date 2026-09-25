"""Stage A: the paper's linear-structure probe.

For every (source layer, target layer, head) cell, fit a single-source ridge map in
content space and record R^2. This is the paper's central evidence -- if cross-model KV
is linearly related, it shows up here -- and it is also the input to cross-layer source
selection in Stage B.

Memory discipline (developed on an 8 GB machine, scaled to 14B -> 32B):
  * each model's KV is captured once to a disk store (`kvtransfer.kvstore`), so each
    model is loaded once and the two are never resident at once
  * activations live only for one shard of sequences, and each derived view is freed
    as soon as it has been consumed
  * statistics live in a few large contiguous tensors (see `batched_gram`) rather than
    one accumulator per cell -- the per-cell version needed 12,768 small allocations and
    fragmented the heap badly enough to force swapping

Source and target may differ in depth. Layer-aligned numbers use proportional alignment
(`select.identity_alignment`), which reduces to l -> l for equal depth.

Output: results/probe_r2*.pt with R^2 of shape (n_src, n_tgt, n_heads) for K_content,
K_rope (layer-aligned diagonal only by default) and V.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from kvtransfer.batched_gram import BatchedGrid, DiagonalGrid
from kvtransfer.kvstore import rope_keys
from kvtransfer.pipeline import calibration_ids, calibration_stores, load_config, paired_batches
from kvtransfer.runtime import output_dir, pair_dims, resolve_runtime
from kvtransfer.select import aligned_cells, identity_alignment

ROOT = Path(__file__).resolve().parents[1]


def per_head(x: torch.Tensor, H: int, D: int) -> torch.Tensor:
    """(L, B, H, S, D) -> (L, H, B*S, D): tokens as observations.

    `permute(...).reshape(...)` cannot return a view here, so this materializes a copy.
    Callers free the source tensor immediately afterwards.
    """
    return x.permute(0, 2, 1, 3, 4).reshape(x.shape[0], H, -1, D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/qwen3_0.6b_to_1.7b.yaml"))
    ap.add_argument("--n-seqs", type=int, default=None, help="override calibration size")
    ap.add_argument("--device", default=None, help="override runtime.device")
    ap.add_argument("--out", default=None, help="default: <output_dir>/probe_r2_n<N>.pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cal, pr = cfg["calibration"], cfg.get("probe", {})
    rt = resolve_runtime(cfg["runtime"], args.device)
    n_seqs = args.n_seqs or cal["n_seqs"]

    from transformers import AutoConfig, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["source"])
    Ls, Lt, H, D = pair_dims(AutoConfig.from_pretrained(cfg["source"]),
                             AutoConfig.from_pretrained(cfg["target"]))
    print(f"src {cfg['source']}: {Ls} layers | tgt {cfg['target']}: {Lt} layers "
          f"| {H} kv heads x {D}")
    print(f"device={rt['device']} dtype={rt['dtype']} shard={rt.get('chunk', 16)} "
          f"stride={cal['stride']}")

    ids = calibration_ids(cfg, tok, n_seqs, ROOT)
    print(f"calibration: {tuple(ids.shape)}")
    t0 = time.time()
    src, tgt = calibration_stores(cfg, rt, ids, ROOT)

    gdt = getattr(torch, pr.get("gram_dtype", "float32"))
    gdev = pr.get("gram_device", "cpu")
    rope_diag = pr.get("rope_diagonal_only", True)
    # Source layer feeding each target layer on the aligned "diagonal".
    align = [identity_alignment(Ls, Lt)[lt][0] for lt in range(Lt)]

    grid_k = BatchedGrid(Ls, Lt, H, D, dtype=gdt, device=gdev)
    grid_v = BatchedGrid(Ls, Lt, H, D, dtype=gdt, device=gdev)
    grid_kr = (
        DiagonalGrid(Lt, H, D, dtype=gdt, device=gdev) if rope_diag
        else BatchedGrid(Ls, Lt, H, D, dtype=gdt, device=gdev)
    )
    nbytes = sum(
        t.numel() * t.element_size()
        for g in (grid_k, grid_v, grid_kr)
        for t in vars(g).values() if isinstance(t, torch.Tensor)
    )
    print(f"statistics: {nbytes/1e9:.2f} GB in contiguous tensors on {gdev}")

    n_chunks = src.meta["n_shards"]
    t1 = time.time()
    for ci, (sk, sv, tk, tv) in enumerate(paired_batches(src, tgt)):
        # Post-RoPE keys, recomputed from the stored content keys at their true positions.
        skr = rope_keys(sk[align] if rope_diag else sk, src, dtype=torch.float16)
        tkr = rope_keys(tk, tgt, dtype=torch.float16)

        # Convert, consume and free one pair at a time: holding all six views at once
        # doubled peak memory for no benefit.
        TK = per_head(tk, H, D).to(gdev); del tk
        SK = per_head(sk, H, D).to(gdev); del sk
        grid_k.update(SK, TK)
        del SK, TK

        TV = per_head(tv, H, D).to(gdev); del tv
        SV = per_head(sv, H, D).to(gdev); del sv
        grid_v.update(SV, TV)
        del SV, TV

        TKR = per_head(tkr, H, D).to(gdev); del tkr
        SKR = per_head(skr, H, D).to(gdev); del skr
        grid_kr.update(SKR, TKR)
        del SKR, TKR

        el = time.time() - t1
        eta = el / (ci + 1) * (n_chunks - ci - 1)
        print(f"  shard {ci+1}/{n_chunks}: n={grid_k.n} obs/head, "
              f"{el/60:.1f}m elapsed, ~{eta/60:.0f}m left", flush=True)

    print(f"\nsolving ... ({(time.time()-t0)/60:.1f} min so far)", flush=True)
    lams = cfg["fit"]["lambdas"]
    lam = lams[len(lams) // 2] if lams else 0.01
    r2 = {
        "k_content": grid_k.solve_r2(lam),      # (Ls, Lt, H)
        "v": grid_v.solve_r2(lam),
    }
    if rope_diag:
        diag = grid_kr.solve_r2(lam)            # (Lt, H), cell (align[lt], lt)
        full = torch.full((Ls, Lt, H), float("nan"))
        for lt in range(Lt):
            full[align[lt], lt] = diag[lt]
        r2["k_rope"] = full
    else:
        r2["k_rope"] = grid_kr.solve_r2(lam)
    r2 = {k: v.cpu() for k, v in r2.items()}

    out = Path(args.out) if args.out else output_dir(cfg, ROOT) / f"probe_r2_n{n_seqs}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "r2": r2, "n_obs": grid_k.n, "n_seqs": n_seqs, "lam": lam,
        "source": cfg["source"], "target": cfg["target"],
        "Ls": Ls, "Lt": Lt, "H": H, "D": D,
    }, out)
    print(f"saved -> {out}  ({(time.time()-t0)/60:.1f} min)")

    print("\n=== Stage A: single-source linear structure ===")
    print(f"observations per head: {grid_k.n}")
    for kind in ("k_content", "k_rope", "v"):
        m = r2[kind].mean(-1)
        finite = m[~torch.isnan(m)]
        if finite.numel() == 0:
            continue
        diag = aligned_cells(m)
        diag = diag[~torch.isnan(diag)]
        print(f"  {kind:10s}: best {float(finite.max()):.3f}   "
              f"layer-aligned mean {float(diag.mean()):.3f}")

    kc, kr = aligned_cells(r2["k_content"].mean(-1)), aligned_cells(r2["k_rope"].mean(-1))
    mask = ~(torch.isnan(kc) | torch.isnan(kr))
    if mask.any():
        print(f"  RoPE stripping: K_content {float(kc[mask].mean()):.3f} "
              f"vs K_rope {float(kr[mask].mean()):.3f}")
    print("\n  paper reference (Qwen3 14B->32B): best single-source K R^2 = 0.81")


if __name__ == "__main__":
    main()
