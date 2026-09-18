"""Stage A: the paper's linear-structure probe.

For every (source layer, target layer, head) cell, fit a single-source OLS/ridge map in
content space and record R^2. This is the paper's central evidence -- if cross-model KV
is linearly related, it shows up here -- and it is also the input to cross-layer source
selection in Stage B.

Memory discipline, for an 8 GB machine:
  * the two models are never resident at once (chunked: capture source, free, capture
    target, free, accumulate, repeat)
  * activations are held only for one chunk of batches, in fp16
  * Gram accumulators are float32 (promoted to float64 inside `solve`), which halves
    the working set to ~1.7 GB for the full 28x28x8 grid

Output: results/probe_r2*.pt with R^2 arrays of shape (n_src, n_tgt, n_heads) for
K_content, K_rope and V.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import yaml

from kvtransfer.capture import KVCapture
from kvtransfer.data import batches, cached_calibration
from kvtransfer.gram import GramAccumulator
from kvtransfer.ridge import solve
from kvtransfer.rope import apply_rope, rope_tables_for
from kvtransfer.stream import subsample

ROOT = Path(__file__).resolve().parents[1]


def load_model(name: str, dtype, device: str):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    if device != "cpu":
        m = m.to(device)
    m.eval()
    return m


def free(*objs):
    for o in objs:
        del o
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


@torch.no_grad()
def capture_chunk(model_name, chunk_ids, cfg_rt, stride, want_rope: bool):
    """Capture content-space (and optionally post-RoPE) KV for a chunk, then free the model.

    Returns (k_content, v, k_rope|None), each (L, n_batches*B, H, S', D) in fp16.
    Hooks move tensors to CPU as they fire, so GPU memory holds only the model and one
    layer's activations regardless of chunk size.
    """
    dtype = getattr(torch, cfg_rt["dtype"])
    device = cfg_rt["device"]
    model = load_model(model_name, dtype, device)
    mc = model.config
    theta = float(mc.rope_parameters["rope_theta"])
    ks, vs, krs = [], [], []
    try:
        for b in chunk_ids:
            b = b.to(device)
            with KVCapture(model, store_dtype=torch.float32) as cap:
                model(input_ids=b, use_cache=False)
                k, v = cap.stacked()          # (L,B,H,S,D) fp32, already on CPU
            if want_rope:
                cos, sin = rope_tables_for(k.shape[3], mc.head_dim, theta, dtype=torch.float32)
                krs.append(subsample(apply_rope(k, cos, sin), stride).half())
            ks.append(subsample(k, stride).half())
            vs.append(subsample(v, stride).half())
            cap.clear()
            free(k, v)
    finally:
        free(model)
    cat = lambda xs: torch.cat(xs, dim=1)
    return cat(ks), cat(vs), (cat(krs) if want_rope else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/qwen3_0.6b_to_1.7b.yaml"))
    ap.add_argument("--n-seqs", type=int, default=None, help="override calibration size")
    ap.add_argument("--out", default=str(ROOT / "results/probe_r2.pt"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cal, rt = cfg["calibration"], cfg["runtime"]
    n_seqs = args.n_seqs or cal["n_seqs"]
    stride, chunk = cal["stride"], rt["chunk"]

    from transformers import AutoConfig, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["source"])
    src_cfg = AutoConfig.from_pretrained(cfg["source"])
    tgt_cfg = AutoConfig.from_pretrained(cfg["target"])
    Ls, Lt = src_cfg.num_hidden_layers, tgt_cfg.num_hidden_layers
    H, D = src_cfg.num_key_value_heads, src_cfg.head_dim
    assert (tgt_cfg.num_key_value_heads, tgt_cfg.head_dim) == (H, D), (
        "not a matched-KV pair: "
        f"src {H}x{D} vs tgt {tgt_cfg.num_key_value_heads}x{tgt_cfg.head_dim}"
    )
    print(f"src {cfg['source']}: {Ls} layers | tgt {cfg['target']}: {Lt} layers | {H} kv heads x {D}")

    print(f"loading {n_seqs} x {cal['seq_len']} FineWeb-Edu sequences ...")
    ids = cached_calibration(
        tok, n_seqs, cal["seq_len"], cal["seed"],
        ROOT / f"results/calib_{n_seqs}x{cal['seq_len']}_s{cal['seed']}.pt",
    )
    print(f"  calibration tensor: {tuple(ids.shape)}")

    # `probe.shards` splits the TARGET layers into groups. With float32 accumulators the
    # whole grid already fits, so this is a safety valve rather than a necessity.
    # Capture is NOT repeated per shard: the same captured chunk updates every shard's
    # Grams before being freed, so each chunk costs exactly one load of each model.
    shards = cfg["probe"]["shards"]
    tgt_groups = [list(range(i, Lt, shards)) for i in range(shards)]

    r2 = {k: torch.zeros(Ls, Lt, H) for k in ("k_content", "k_rope", "v")}
    n_obs = 0
    t0 = time.time()

    rope_diag = cfg["probe"].get("rope_diagonal_only", True)
    # float32 accumulators halve the working set. `solve` promotes to float64 before the
    # cancellation-prone centering step, and at this scale the R^2 difference measured
    # under 1e-6 -- immaterial for a diagnostic, and it keeps the machine off swap.
    gdt = getattr(torch, cfg["probe"].get("gram_dtype", "float32"))
    all_grams = []
    for tgt_layers in tgt_groups:
        g = {kind: [[[GramAccumulator(D, D, dtype=gdt) for _ in range(H)] for _ in tgt_layers]
                    for _ in range(Ls)]
             for kind in ("k_content", "v")}
        # K_rope exists to answer one question -- does stripping RoPE help? -- so when
        # `rope_diagonal_only` it is fitted only where src layer == tgt layer.
        g["k_rope"] = (
            {lt: [GramAccumulator(D, D, dtype=gdt) for _ in range(H)] for lt in tgt_layers}
            if rope_diag
            else [[[GramAccumulator(D, D, dtype=gdt) for _ in range(H)] for _ in tgt_layers]
                  for _ in range(Ls)]
        )
        all_grams.append(g)

    n_chunks = (ids.shape[0] + chunk - 1) // chunk
    for ci, c0 in enumerate(range(0, ids.shape[0], chunk)):
        chunk_ids = list(batches(ids[c0 : c0 + chunk], rt["batch_size"]))
        sk, sv, skr = capture_chunk(cfg["source"], chunk_ids, rt, stride, want_rope=True)
        tk, tv, tkr = capture_chunk(cfg["target"], chunk_ids, rt, stride, want_rope=True)

        per_head = lambda x: x.permute(0, 2, 1, 3, 4).reshape(x.shape[0], H, -1, D)
        SK, SV, SKR = per_head(sk), per_head(sv), per_head(skr)
        TK, TV, TKR = per_head(tk), per_head(tv), per_head(tkr)

        # Every shard consumes this chunk before it is freed: capture cost is paid once.
        for grams, tgt_layers in zip(all_grams, tgt_groups):
            for ls in range(Ls):
                for j, lt in enumerate(tgt_layers):
                    for h in range(H):
                        grams["k_content"][ls][j][h].update(SK[ls, h], TK[lt, h])
                        grams["v"][ls][j][h].update(SV[ls, h], TV[lt, h])
                        if not rope_diag:
                            grams["k_rope"][ls][j][h].update(SKR[ls, h], TKR[lt, h])
            if rope_diag:
                for lt in tgt_layers:
                    if lt < Ls:                      # layer-aligned diagonal
                        for h in range(H):
                            grams["k_rope"][lt][h].update(SKR[lt, h], TKR[lt, h])

        n_obs = all_grams[0]["k_content"][0][0][0].n
        el = time.time() - t0
        eta = el / (ci + 1) * (n_chunks - ci - 1)
        print(f"  chunk {ci+1}/{n_chunks}: {min(c0+chunk, ids.shape[0])}/{ids.shape[0]} seqs, "
              f"n={n_obs} obs, {el/60:.1f}m elapsed, ~{eta/60:.0f}m left", flush=True)
        free(sk, sv, skr, tk, tv, tkr, SK, SV, SKR, TK, TV, TKR)

    print("\nsolving ...", flush=True)
    r2["k_rope"].fill_(float("nan"))     # cells never fitted stay NaN, not 0
    for grams, tgt_layers in zip(all_grams, tgt_groups):
        for kind in ("k_content", "v"):
            for ls in range(Ls):
                for j, lt in enumerate(tgt_layers):
                    for h in range(H):
                        r2[kind][ls, lt, h] = solve(grams[kind][ls][j][h], lam=0.01).r2
        if rope_diag:
            for lt in tgt_layers:
                if lt < Ls:
                    for h in range(H):
                        r2["k_rope"][lt, lt, h] = solve(grams["k_rope"][lt][h], lam=0.01).r2
        else:
            for ls in range(Ls):
                for j, lt in enumerate(tgt_layers):
                    for h in range(H):
                        r2["k_rope"][ls, lt, h] = solve(grams["k_rope"][ls][j][h], lam=0.01).r2
    free(all_grams)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "r2": r2, "n_obs": n_obs, "n_seqs": n_seqs,
        "source": cfg["source"], "target": cfg["target"],
        "Ls": Ls, "Lt": Lt, "H": H, "D": D,
    }, out)

    print(f"\nsaved -> {out}  ({(time.time()-t0)/60:.1f} min total)")
    print("\n=== Stage A: single-source linear structure ===")
    print(f"observations per head: {n_obs}")
    for kind in ("k_content", "k_rope", "v"):
        m = r2[kind].mean(-1)            # head-averaged (Ls, Lt)
        diag = m.diagonal()
        if torch.isnan(m).all():
            continue
        finite = m[~torch.isnan(m)]
        best = finite.max().item()
        if not torch.isnan(m).any():
            bi = torch.unravel_index(m.argmax(), m.shape)
            where = f"(src {bi[0].item()} -> tgt {bi[1].item()})"
        else:
            where = "(diagonal only)"
        print(f"  {kind:10s}: best R^2 = {best:.3f} {where}, "
              f"layer-aligned mean = {diag[~torch.isnan(diag)].mean():.3f}")

    kc, kr = r2["k_content"].mean(-1).diagonal(), r2["k_rope"].mean(-1).diagonal()
    mask = ~(torch.isnan(kc) | torch.isnan(kr))
    if mask.any():
        print(f"\n  RoPE stripping effect (layer-aligned): "
              f"K_content {kc[mask].mean():.3f} vs K_rope {kr[mask].mean():.3f}")
    print(f"\n  paper reference (Qwen3 14B->32B): best single-source K_stripped R^2 = 0.81")


if __name__ == "__main__":
    main()
