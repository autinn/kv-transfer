# Cross-Model KV Cache Transfer — Qwen3 0.6B → 1.7B

A replication of **"Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear
Mapping for Prefill Reuse"** ([arXiv 2608.03893](https://arxiv.org/html/2608.03893v1),
Heo et al., NVIDIA), tested one scale below anything the paper reports.

## The claim under test

Within a model family, a small model's KV cache can be mapped into a large model's
expected format by a **per-head ridge regression fitted in closed form**, letting the
large model skip prefill entirely. The paper reports 73–98% accuracy retention at
2.7–25× speedup.

Every pair in the paper is ≥3B parameters. The open question this repo asks is whether
the effect survives at sub-2B scale, where the two models have had far less opportunity
to converge on similar internal representations.

## Why Qwen3 0.6B → 1.7B

The method requires a **matched-KV** pair: identical KV head count and per-head
dimension. Verified from the published configs:

|                | Qwen3-0.6B | Qwen3-1.7B |
|----------------|-----------|-----------|
| layers         | 28        | 28        |
| KV heads       | 8         | 8         |
| head_dim       | 128       | 128       |
| hidden         | 1024      | 2048      |
| rope_theta     | 1e6       | 1e6       |
| tokenizer      | Qwen2Tokenizer | identical |

Both are dense full-attention, the tokenizer is shared (required — the regression pairs
KV vectors *token by token*, so both models must see identical token streams), and the
2.8× parameter ratio sits inside the paper's 2.3–8.8× range.

Equal depth is a bonus no pair in the paper has: it removes the layer-alignment
confound and gives a clean identity-alignment baseline (layer *l* → layer *l*) to test
whether cross-layer source selection earns its complexity.

### Why not Qwen3.5 0.8B → 2B

Ruled out after reading their configs, and recorded here so the negative result is not
rediscovered. Qwen3.5-0.8B and 2B are **hybrid linear-attention** models:
`full_attention_interval: 4` means only 6 of 24 layers are full attention; the other 18
carry recurrent state rather than a KV cache. They are also 2 KV heads × head_dim 256,
so they are not matched-KV against Qwen3. The paper scopes explicitly to dense
full-attention, and end-to-end retention is not measurable when three quarters of the
layers have no cache to transfer.

## Method

Per target layer *l* and KV head *h*, fitted in **content space** (RoPE stripped):

```
K̂_t = (K_s · R_s⁻¹(t) · W_K + b_K) · R_t(t)
V̂_t =  V_s · W_V + b_V

W* = (XᵀX + λI)⁻¹ XᵀY            λ = 0.01
X^l = [K_s^(l₁) ‖ … ‖ K_s^(l_k)]  top-k source layers by head-averaged R²
```

Two details that are easy to get wrong and fatal if you do:

- **RoPE convention.** Qwen3 uses non-interleaved (GPT-NeoX `rotate_half`), not the
  interleaved GPT-J form. The wrong one still runs and still produces plausible numbers.
  `tests/test_rope.py` checks ours against HF's own `apply_rotary_pos_emb`.
- **Hook placement.** Qwen3 applies RMSNorm to keys *before* RoPE, so content space is
  the output of `k_norm`, not of `k_proj` and not the cache (which stores post-RoPE
  keys). Verified: capture + re-rotation reproduces the model's own cache with **0.0
  error** across all 28 layers.

**Caveat recorded up front:** because both models share `rope_theta` and `head_dim`,
`R_s == R_t` for this pair, so strip-then-re-apply cancels at the calibration positions.
Stripping still earns its place — it makes the fitted `W` position-independent, which is
what lets a mapper fitted at positions 0–1023 work at position 4096 — but this pair does
**not** validate the cross-RoPE-config part of the paper's contribution.

## Guardrail

`tests/test_identity.py` fits a mapper from the 1.7B **to itself** and grafts the result
back. It must reproduce standalone logits. This catches nearly every silent failure
mode — wrong RoPE convention, wrong hook site, GQA head misalignment, position-id
off-by-one, cache layout transposition — and it runs first, because no downstream number
means anything until it passes.

Measured: grafted logits match standalone to **2.9e-05**.

## Running

```bash
uv venv && uv pip install -e ".[dev]"
source .env.sh                     # disables the HF Xet cache (see Notes)

uv run pytest tests/ -v            # 14 fast tests
uv run pytest tests/ -v -m slow    # + 4 end-to-end guardrails (needs 0.6B weights)

uv run python scripts/02_probe_r2.py --n-seqs 50    # Stage A: linear-structure probe
uv run python scripts/03_fit.py --k 1 4 8           # Stage B: fit production mappers
uv run python scripts/05_eval.py                    # prefix-conditioned perplexity
```

## Notes on running this on a small machine

Developed on an M1 with 8 GB of unified memory, which shaped several decisions:

- **Never materialize the design matrix.** With ~128K observations per head and up to
  3584 features, the design matrix would be gigabytes. Streaming `XᵀX`/`XᵀY`
  accumulation makes it kilobytes. This is what makes the paper's closed form practical
  on a laptop.
- **Never hold both models resident.** Capture from one, free it, then load the other.
- **Gram accumulators are float32**, promoted to float64 inside `solve()` before the
  cancellation-prone centering step. Measured R² difference at this scale is under
  1e-6, and it halves the working set from 3.4 GB to 1.7 GB. An earlier float64 run
  drove an already-swapped machine into uninterruptible sleep.
- **`HF_HUB_DISABLE_XET=1`** (in `.env.sh`). Hugging Face's Xet chunk store keeps a
  second copy of every downloaded blob; here it consumed 2.6 GB while the visible cache
  directory reported 4.3 MB.
- **No `shuffle(buffer_size=...)` on the streaming dataset.** The buffer must fill
  before yielding anything: 12 minutes and 270 MB for a 4-sequence smoke test, versus
  6.8 seconds without it.

## Status

| Phase | State |
|-------|-------|
| 0 — verified math core | done, 18 tests green |
| 1 — calibration data | done |
| 2 — R² linear-structure probe | running |
| 3 — fit + k-sweep | code ready |
| 4 — perplexity + retention | code ready |
| 5 — writeup | pending |

Results land in `results/` as they are produced. **A faithful negative result is a
success**: if sub-2B models do not share enough linear structure, that is a real finding
about the scale-dependence of the paper's claim, and this repo is built to measure it
rather than to confirm it.
