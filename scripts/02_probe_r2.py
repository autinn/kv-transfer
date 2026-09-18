"""Stage A: the paper's linear-structure probe.

For every (source layer, target layer, head) cell, fit a single-source ridge map in
content space and record R^2. This is the paper's central evidence -- if cross-model KV
is linearly related, it shows up here -- and it is also the input to cross-layer source
selection in Stage B.

Memory discipline (developed on an 8 GB machine):
  * the two models are never resident at once: capture from the source, free it, then
    capture from the target
  * activations live only for one chunk of sequences, in fp16, and each derived view is
    freed as soon as it has been consumed
  * statistics live in a few large contiguous tensors (see `batched_gram`) rather than
    one accumulator per cell -- the per-cell version needed 12,768 small allocations and
    fragmented the heap badly enough to force swapping

Output: results/probe_r2*.pt with R^2 of shape (n_src, n_tgt, n_heads) for K_content,
K_rope (layer-aligned diagonal only by default) and V.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch
import yaml

from kvtransfer.batched_gram import BatchedGrid, DiagonalGrid
from kvtransfer.capture import KVCapture
from kvtransfer.data import batches, cached_calibration
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


def per_head(x: torch.Tensor, H: int, D: int) -> torch.Tensor:
    """(L, B, H, S, D) -> (L, H, B*S, D): tokens as observations.

    `permute(...).reshape(...)` cannot return a view here, so this materializes a copy.
    Callers free the source tensor immediately afterwards.
    """
    return x.permute(0, 2, 1, 3, 4).reshape(x.shape[0], H, -1, D)


@torch.no_grad()
def capture_chunk(model_name, chunk_ids, rt, stride, want_rope: bool):
    """Capture content-space (and optionally post-RoPE) KV for a chunk, then free the model.

    Returns (k_content, v, k_rope|None), each (L, n_seqs, H, S', D) in fp16 on CPU.
    """
    model = load_model(model_name, getattr(torch, rt["dtype"]), rt["device"])
    mc = model.config
    theta = float(mc.rope_parameters["rope_theta"])
    ks, vs, krs = [], [], []
    try:
        for b in chunk_ids:
            with KVCapture(model, store_dtype=torch.float32) as cap:
                model(input_ids=b.to(rt["device"]), use_cache=False)
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
    ap.add_argument("--device", default=None, help="override runtime.device")
    ap.add_argument("--out", default=str(ROOT / "results/probe_r2.pt"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cal, rt, pr = cfg["calibration"], cfg["runtime"], cfg["probe"]
    if args.device:
        rt["device"] = args.device
        if args.device == "cpu" and rt["dtype"] == "float16":
            rt["dtype"] = "float32"          # fp16 matmuls are not supported on CPU
    n_seqs = args.n_seqs or cal["n_seqs"]
    stride, chunk = cal["stride"], rt["chunk"]

    from transformers import AutoConfig, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["source"])
    scfg = AutoConfig.from_pretrained(cfg["source"])
    tcfg = AutoConfig.from_pretrained(cfg["target"])
    Ls, Lt = scfg.num_hidden_layers, tcfg.num_hidden_layers
    H, D = scfg.num_key_value_heads, scfg.head_dim
    assert (tcfg.num_key_value_heads, tcfg.head_dim) == (H, D), (
        f"not a matched-KV pair: src {H}x{D} vs "
        f"tgt {tcfg.num_key_value_heads}x{tcfg.head_dim}"
    )
    print(f"src {cfg['source']}: {Ls} layers | tgt {cfg['target']}: {Lt} layers "
          f"| {H} kv heads x {D}")
    print(f"device={rt['device']} dtype={rt['dtype']} chunk={chunk} stride={stride}")

    ids = cached_calibration(
        tok, n_seqs, cal["seq_len"], cal["seed"],
        ROOT / f"results/calib_{n_seqs}x{cal['seq_len']}_s{cal['seed']}.pt",
    )
    print(f"calibration: {tuple(ids.shape)}")

    gdt = getattr(torch, pr.get("gram_dtype", "float32"))
    rope_diag = pr.get("rope_diagonal_only", True)

    grid_k = BatchedGrid(Ls, Lt, H, D, dtype=gdt)
    grid_v = BatchedGrid(Ls, Lt, H, D, dtype=gdt)
    grid_kr = (
        DiagonalGrid(min(Ls, Lt), H, D, dtype=gdt) if rope_diag
        else BatchedGrid(Ls, Lt, H, D, dtype=gdt)
    )
    nbytes = sum(
        t.numel() * t.element_size()
        for g in (grid_k, grid_v, grid_kr)
        for t in vars(g).values() if isinstance(t, torch.Tensor)
    )
    print(f"statistics: {nbytes/1e9:.2f} GB in contiguous tensors")

    n_chunks = (ids.shape[0] + chunk - 1) // chunk
    t0 = time.time()
    for ci, c0 in enumerate(range(0, ids.shape[0], chunk)):
        chunk_ids = list(batches(ids[c0 : c0 + chunk], rt["batch_size"]))

        sk, sv, skr = capture_chunk(cfg["source"], chunk_ids, rt, stride, want_rope=True)
        tk, tv, tkr = capture_chunk(cfg["target"], chunk_ids, rt, stride, want_rope=True)

        # Convert, consume and free one pair at a time: holding all six views at once
        # doubled peak memory for no benefit.
        TK = per_head(tk, H, D); free(tk)
        SK = per_head(sk, H, D); free(sk)
        grid_k.update(SK, TK)
        free(SK, TK)

        TV = per_head(tv, H, D); free(tv)
        SV = per_head(sv, H, D); free(sv)
        grid_v.update(SV, TV)
        free(SV, TV)

        TKR = per_head(tkr, H, D); free(tkr)
        SKR = per_head(skr, H, D); free(skr)
        if rope_diag:
            L = min(Ls, Lt)
            grid_kr.update(SKR[:L], TKR[:L])
        else:
            grid_kr.update(SKR, TKR)
        free(SKR, TKR)

        el = time.time() - t0
        eta = el / (ci + 1) * (n_chunks - ci - 1)
        print(f"  chunk {ci+1}/{n_chunks}: {min(c0+chunk, ids.shape[0])}/{ids.shape[0]} seqs, "
              f"n={grid_k.n} obs/head, {el/60:.1f}m elapsed, ~{eta/60:.0f}m left", flush=True)

    print("\nsolving ...", flush=True)
    lams = cfg["fit"]["lambdas"]
    lam = lams[len(lams) // 2] if lams else 0.01
    r2 = {
        "k_content": grid_k.solve_r2(lam),      # (Ls, Lt, H)
        "v": grid_v.solve_r2(lam),
    }
    if rope_diag:
        diag = grid_kr.solve_r2(lam)            # (L, H)
        full = torch.full((Ls, Lt, H), float("nan"))
        for l in range(diag.shape[0]):
            full[l, l] = diag[l]
        r2["k_rope"] = full
    else:
        r2["k_rope"] = grid_kr.solve_r2(lam)

    out = Path(args.out)
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
        diag = m.diagonal()
        diag = diag[~torch.isnan(diag)]
        print(f"  {kind:10s}: best {float(finite.max()):.3f}   "
              f"layer-aligned mean {float(diag.mean()):.3f}")

    kc, kr = r2["k_content"].mean(-1).diagonal(), r2["k_rope"].mean(-1).diagonal()
    mask = ~(torch.isnan(kc) | torch.isnan(kr))
    if mask.any():
        print(f"  RoPE stripping: K_content {float(kc[mask].mean()):.3f} "
              f"vs K_rope {float(kr[mask].mean()):.3f}")
    print("\n  paper reference (Qwen3 14B->32B): best single-source K R^2 = 0.81")


if __name__ == "__main__":
    main()
