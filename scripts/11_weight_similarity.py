"""Cosine similarity of k_proj weights for every (source layer, target layer) pair.

Two independently trained models have unrelated weights, so their cosine is ~0. A target
layer that was initialised from a source layer keeps a clearly non-zero cosine with it
even after further training. Only pairs with the same hidden size can be compared this
way (k_proj is kv_heads*head_dim x hidden_size).

Reads only the k_proj tensors from the Hub with byte-range requests (~1 GB for
14B -> 32B), so the full checkpoints are never downloaded.

    python scripts/11_weight_similarity.py --source Qwen/Qwen3-14B --target Qwen/Qwen3-32B
"""

import argparse
import json
import struct
from pathlib import Path

import torch
from huggingface_hub import HfFileSystem, hf_hub_download

ROOT = Path(__file__).resolve().parent.parent


def k_proj_weights(repo: str) -> torch.Tensor:
    """(n_layers, kv_heads*head_dim*hidden) float32, one flattened k_proj per layer."""
    cfg = json.load(open(hf_hub_download(repo, "config.json")))
    wmap = json.load(open(hf_hub_download(repo, "model.safetensors.index.json")))["weight_map"]
    fs, headers, out = HfFileSystem(), {}, []
    for layer in range(cfg["num_hidden_layers"]):
        name = f"model.layers.{layer}.self_attn.k_proj.weight"
        path = f"{repo}/{wmap[name]}"
        with fs.open(path, "rb", block_size=1 << 20) as f:
            if path not in headers:  # safetensors: u64 header length, JSON header, data
                n = struct.unpack("<Q", f.read(8))[0]
                headers[path] = (8 + n, json.loads(f.read(n)))
            base, header = headers[path]
            assert header[name]["dtype"] == "BF16", header[name]["dtype"]
            a, b = header[name]["data_offsets"]
            f.seek(base + a)
            raw = f.read(b - a)
        out.append(torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).float())
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Qwen/Qwen3-14B")
    ap.add_argument("--target", default="Qwen/Qwen3-32B")
    ap.add_argument("--out", type=Path, default=ROOT / "results/qwen3_14b_to_32b/kproj_cos.json")
    args = ap.parse_args()

    src, tgt = k_proj_weights(args.source), k_proj_weights(args.target)
    if src.shape[1] != tgt.shape[1]:
        raise SystemExit(f"k_proj shapes differ ({src.shape[1]} vs {tgt.shape[1]} elements): "
                         "different hidden sizes, no direct weight comparison")
    src = src / src.norm(dim=1, keepdim=True)
    tgt = tgt / tgt.norm(dim=1, keepdim=True)
    cos = src @ tgt.T  # (Ls, Lt)

    best = cos.argmax(0)
    print("best source layer per target layer:", best.tolist())
    print("cosine at that layer:", [round(x, 2) for x in cos.max(0).values.tolist()])
    print(f"median cosine over all pairs: {cos.median():.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "source": args.source, "target": args.target, "weight": "self_attn.k_proj",
        "cos": cos.round(decimals=4).tolist(), "best_src": best.tolist(),
    }))
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
