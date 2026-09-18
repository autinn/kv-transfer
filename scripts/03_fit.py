"""Stage B: fit production mappers at a chosen k, using Stage A's source selection.

Reads results/probe_r2*.pt, picks the top-k source layers per target layer, streams the
calibration set once more to accumulate top-k Grams, and solves for the per-head ridge
maps. Saves a KVMapper per (k, lambda).

Because the Gram is cached per k, sweeping lambda afterwards is one Cholesky each --
the expensive part is the streaming pass, so all lambdas are solved from the same pass.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml

from kvtransfer.data import batches, cached_calibration
from kvtransfer.mapper import KVMapper
from kvtransfer.ridge import solve
from kvtransfer.select import identity_alignment, top_k
from kvtransfer.stream import TopKGrams

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

_probe = import_module("02_probe_r2")
capture_chunk, free = _probe.capture_chunk, _probe.free

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/qwen3_0.6b_to_1.7b.yaml"))
    ap.add_argument("--probe", default=str(ROOT / "results/probe_r2_n100.pt"))
    ap.add_argument("--n-seqs", type=int, default=None)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--identity-baseline", action="store_true",
                    help="also fit the layer-aligned single-source baseline")
    ap.add_argument("--outdir", default=str(ROOT / "results/mappers"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cal, rt = cfg["calibration"], cfg["runtime"]
    probe = torch.load(args.probe)
    r2_k, r2_v = probe["r2"]["k_content"], probe["r2"]["v"]
    Ls, Lt, H, D = probe["Ls"], probe["Lt"], probe["H"], probe["D"]
    n_seqs = args.n_seqs or probe["n_seqs"]

    from transformers import AutoConfig, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["source"])
    theta_t = float(AutoConfig.from_pretrained(cfg["target"]).rope_parameters["rope_theta"])

    ids = cached_calibration(
        tok, n_seqs, cal["seq_len"], cal["seed"],
        ROOT / f"results/calib_{n_seqs}x{cal['seq_len']}_s{cal['seed']}.pt",
    )
    print(f"calibration {tuple(ids.shape)}")

    # Selections to fit. Keys and values get their own selections: the probe shows they
    # do not necessarily peak at the same source layers.
    selections = {}
    for k in args.k:
        selections[f"k{k}"] = (top_k(r2_k, k), top_k(r2_v, k))
    if args.identity_baseline:
        ident = identity_alignment(Ls, Lt)
        selections["identity"] = (ident, ident)

    lams = cfg["fit"]["lambdas"]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for name, (sel_k, sel_v) in selections.items():
        print(f"\n=== fitting {name} ===")
        sizes = {len(v) for v in sel_k.values()}
        print(f"  sources per target layer: {sorted(sizes)}")
        # Keys and values may select different source layers, so each needs its own
        # feature layout. When the selections coincide (the common case) one Gram set
        # carries both and the second pass is skipped.
        same_sel = sel_k == sel_v
        gk = TopKGrams(sel_k, H, D)
        gv = gk if same_sel else TopKGrams(sel_v, H, D)

        t0 = time.time()
        chunk = rt["chunk"]
        n_chunks = (ids.shape[0] + chunk - 1) // chunk
        for ci, c0 in enumerate(range(0, ids.shape[0], chunk)):
            chunk_ids = list(batches(ids[c0 : c0 + chunk], rt["batch_size"]))
            sk, sv, _ = capture_chunk(cfg["source"], chunk_ids, rt, cal["stride"], want_rope=False)
            tk, tv, _ = capture_chunk(cfg["target"], chunk_ids, rt, cal["stride"], want_rope=False)
            gk.update(sk, sv, tk, tv)
            if not same_sel:
                gv.update(sk, sv, tk, tv)
            print(f"  chunk {ci+1}/{n_chunks}, {(time.time()-t0)/60:.1f}m", flush=True)
            free(sk, sv, tk, tv)

        for lam in lams:
            mapper = KVMapper(sel_k, D, H, theta_t)
            r2s_k, r2s_v = [], []
            for lt in range(Lt):
                for h in range(H):
                    sk_sol = solve(gk.K[(lt, h)], lam=lam)
                    sv_sol = solve(gv.V[(lt, h)], lam=lam)
                    mapper.set("K", lt, h, sk_sol.W, sk_sol.b, sk_sol.r2)
                    mapper.set("V", lt, h, sv_sol.W, sv_sol.b, sv_sol.r2)
                    r2s_k.append(sk_sol.r2)
                    r2s_v.append(sv_sol.r2)
            out = outdir / f"mapper_{name}_lam{lam}.pt"
            torch.save(mapper.state_dict(), out)
            mk = sum(r2s_k) / len(r2s_k)
            mv = sum(r2s_v) / len(r2s_v)
            print(f"  lam={lam}: mean R^2  K={mk:.3f}  V={mv:.3f}  -> {out.name}")
        free(gk, gv)


if __name__ == "__main__":
    main()
