"""Stage B: fit production mappers at a chosen k, using Stage A's source selection.

Reads the probe, picks the top-k source layers per target layer, streams the captured
calibration KV back from disk to accumulate top-k Grams, and solves for the per-head
ridge maps. Saves a KVMapper per (k, lambda).

Capture happens once (`kvtransfer.kvstore`); if the probe already ran on the same
calibration set, this stage never loads a model. Each (k, lambda) is then a disk pass
plus one Cholesky per head, and all lambdas are solved from the same pass.

Large pairs do not fit every target layer's Grams in memory at once (14B -> 32B at k=20
is ~50 GB of fp64), so target layers are fitted in groups under `fit.max_gram_gb`, one
disk pass per group.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from kvtransfer.mapper import KVMapper
from kvtransfer.pipeline import calibration_ids, calibration_stores, load_config, paired_batches
from kvtransfer.ridge import solve
from kvtransfer.runtime import output_dir, pair_dims, resolve_runtime, rope_theta
from kvtransfer.select import identity_alignment, top_k
from kvtransfer.stream import TopKGrams

ROOT = Path(__file__).resolve().parents[1]


def layer_groups(sel_k, sel_v, H, D, budget_bytes, all_heads: bool = False) -> list[list[int]]:
    """Split target layers into groups whose Grams fit under the budget."""
    same = sel_k == sel_v
    groups, cur, used = [], [], 0
    for lt in sorted(sel_k):
        if same:
            need = TopKGrams.bytes_per_target_layer(len(sel_k[lt]), H, D, n_kinds=2, all_heads=all_heads)
        else:
            need = (TopKGrams.bytes_per_target_layer(len(sel_k[lt]), H, D, n_kinds=1, all_heads=all_heads)
                    + TopKGrams.bytes_per_target_layer(len(sel_v[lt]), H, D, n_kinds=1, all_heads=all_heads))
        if cur and used + need > budget_bytes:
            groups.append(cur)
            cur, used = [], 0
        cur.append(lt)
        used += need
    if cur:
        groups.append(cur)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/qwen3_0.6b_to_1.7b.yaml"))
    ap.add_argument("--probe", default=None, help="default: <output_dir>/probe_r2_n<N>.pt")
    ap.add_argument("--n-seqs", type=int, default=None)
    ap.add_argument("--device", default=None, help="override runtime.device")
    ap.add_argument("--k", type=int, nargs="+", default=None, help="default: fit.k_sweep")
    ap.add_argument("--identity-baseline", action="store_true",
                    help="also fit the layer-aligned single-source baseline")
    ap.add_argument("--all-heads", action="store_true",
                    help="feature = all source KV heads per selected layer (k*kv_heads*head_dim), "
                         "matching the paper's stated construction, instead of the same-numbered "
                         "source head only (k*head_dim, this repo's original default)")
    ap.add_argument("--outdir", default=None, help="default: <output_dir>/mappers")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cal, fit = cfg["calibration"], cfg.get("fit", {})
    rt = resolve_runtime(cfg["runtime"], args.device)
    out_root = output_dir(cfg, ROOT)
    n_seqs = args.n_seqs or cal["n_seqs"]
    probe_path = Path(args.probe) if args.probe else out_root / f"probe_r2_n{n_seqs}.pt"
    probe = torch.load(probe_path)
    r2_k, r2_v = probe["r2"]["k_content"], probe["r2"]["v"]

    from transformers import AutoConfig, AutoTokenizer

    scfg, tcfg = AutoConfig.from_pretrained(cfg["source"]), AutoConfig.from_pretrained(cfg["target"])
    Ls, Lt, H, D = pair_dims(scfg, tcfg)
    if (probe["Ls"], probe["Lt"], probe["H"], probe["D"]) != (Ls, Lt, H, D):
        raise ValueError(f"{probe_path} was run on a different model pair "
                         f"({probe['source']} -> {probe['target']})")
    theta_t = rope_theta(tcfg)

    tok = AutoTokenizer.from_pretrained(cfg["source"])
    ids = calibration_ids(cfg, tok, n_seqs, ROOT)
    print(f"calibration {tuple(ids.shape)}  | {Ls} -> {Lt} layers, {H}x{D}")
    src, tgt = calibration_stores(cfg, rt, ids, ROOT)

    # Selections to fit. Keys and values get their own selections: the probe shows they
    # do not necessarily peak at the same source layers.
    selections = {}
    for k in args.k or fit.get("k_sweep", [1, 4, 8]):
        if k > Ls:
            print(f"skipping k={k}: source has only {Ls} layers")
            continue
        selections[f"k{k}"] = (top_k(r2_k, k), top_k(r2_v, k))
    if args.identity_baseline:
        ident = identity_alignment(Ls, Lt)
        selections["identity"] = (ident, ident)

    lams = fit.get("lambdas", [0.01])
    budget = float(fit.get("max_gram_gb", 8)) * 1e9
    gdev = fit.get("gram_device", "cpu")
    outdir = Path(args.outdir) if args.outdir else out_root / "mappers"
    outdir.mkdir(parents=True, exist_ok=True)

    for name, (sel_k, sel_v) in selections.items():
        print(f"\n=== fitting {name} ===")
        print(f"  sources per target layer: {sorted({len(v) for v in sel_k.values()})}")
        same_sel = sel_k == sel_v
        groups = layer_groups(sel_k, sel_v, H, D, budget, all_heads=args.all_heads)
        print(f"  {len(groups)} target-layer group(s) under {budget/1e9:.0f} GB"
              + (" (all_heads)" if args.all_heads else ""))
        # The mapper must record BOTH selections: V maps fitted on sel_v's features are
        # meaningless if applied to sel_k's layers at inference.
        mappers = {lam: KVMapper(sel_k, D, H, theta_t, n_src_layers=Ls, src_layers_v=sel_v,
                                 all_heads=args.all_heads)
                   for lam in lams}
        if not same_sel:
            n_diff = sum(sel_k[lt] != sel_v[lt] for lt in sel_k)
            print(f"  K and V select different source layers in {n_diff}/{Lt} target layers")
        t0 = time.time()
        for gi, group in enumerate(groups):
            gk = TopKGrams({lt: sel_k[lt] for lt in group}, H, D, device=gdev,
                           kinds=("K", "V") if same_sel else ("K",), all_heads=args.all_heads)
            gv = gk if same_sel else TopKGrams({lt: sel_v[lt] for lt in group}, H, D,
                                               device=gdev, kinds=("V",), all_heads=args.all_heads)
            for sk, sv, tk, tv in paired_batches(src, tgt):
                if gdev != "cpu":
                    sk, sv, tk, tv = (x.to(gdev) for x in (sk, sv, tk, tv))
                gk.update(sk, sv, tk, tv)
                if not same_sel:
                    gv.update(sk, sv, tk, tv)
            for lam in lams:
                for lt in group:
                    for h in range(H):
                        sk_sol = solve(gk.K[(lt, h)], lam=lam)
                        sv_sol = solve(gv.V[(lt, h)], lam=lam)
                        mappers[lam].set("K", lt, h, sk_sol.W, sk_sol.b, sk_sol.r2)
                        mappers[lam].set("V", lt, h, sv_sol.W, sv_sol.b, sv_sol.r2)
            del gk, gv
            print(f"  group {gi+1}/{len(groups)} (target layers {group[0]}-{group[-1]}) "
                  f"{(time.time()-t0)/60:.1f}m", flush=True)

        suffix = "_allheads" if args.all_heads else ""
        for lam, mapper in mappers.items():
            mapper.check_compat(Ls, Lt, H, D)
            out = outdir / f"mapper_{name}_lam{lam}{suffix}.pt"
            torch.save(mapper.state_dict(), out)
            print(f"  lam={lam}: mean R^2  K={mapper.mean_r2('K'):.3f}  "
                  f"V={mapper.mean_r2('V'):.3f}  -> {out.name}")


if __name__ == "__main__":
    main()
