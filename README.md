# Cross-Model KV Cache Transfer — Qwen3 0.6B → 1.7B

A replication of **"Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear
Mapping for Prefill Reuse"** ([arXiv 2608.03893](https://arxiv.org/html/2608.03893v1),
Heo et al., NVIDIA), tested one scale below anything the paper reports.

**[REPORT.md](REPORT.md)** — what was measured and verified.
**[NEXT_STEPS.md](NEXT_STEPS.md)** — what to run on a rented GPU, and which GPU to rent.

---

## The claim under test

Within a model family, a small model's KV cache can be mapped into a large model's
expected format by a **per-head ridge regression fitted in closed form**, letting the
large model skip prefill entirely. The paper reports 73–98% accuracy retention at
2.7–25× speedup.

Every pair in the paper is ≥3B parameters. The question this repo asks: **does the
effect survive at sub-2B scale**, where two models have had far less opportunity to
converge on similar internal representations?

A negative answer is a real result. Nothing below 3B has been reported.

---

## Method

Per target layer *l* and KV head *h*, fitted in **content space** (RoPE stripped):

```
K̂_t = (K_s · R_s⁻¹(t) · W_K + b_K) · R_t(t)      keys: un-rotate, map, re-rotate
V̂_t =  V_s · W_V + b_V                            values carry no position

W* = (XᵀX + λI)⁻¹ XᵀY                             λ = 0.01
X^l = [K_s^(l₁) ‖ … ‖ K_s^(l_k)]                  top-k source layers by head-averaged R²
```

The pipeline mirrors the paper's own two-stage structure:

**Stage A — linear-structure probe.** Fit a single-source map for every
(source layer, target layer, head) cell and record R². This is the paper's central
evidence *and* the input to source selection, so it is run first and is cheap.

**Stage B — production fit.** Using Stage A's ranking, concatenate the top-k source
layers per target layer and fit the real mappers. One streaming pass; every λ afterwards
is a single Cholesky on a cached Gram.

### Two details that are easy to get wrong

- **RoPE convention.** Qwen3 uses non-interleaved (GPT-NeoX `rotate_half`), pairing
  dimension *i* with *i + D/2* — not the interleaved GPT-J form. The wrong one still
  runs and still produces plausible numbers while corrupting every key.
  `tests/test_rope.py` checks against HF's own `apply_rotary_pos_emb`.
- **Hook placement.** Qwen3 applies RMSNorm to keys *before* RoPE, so content space is
  the output of `k_norm` — not `k_proj`, and not the cache (which stores post-RoPE keys).
  Verified: capture + re-rotation reproduces the model's own cache with **0.0 error**.

---

## Why this model pair

The method needs a **matched-KV** pair: same KV head count, same per-head dimension,
dense full attention, shared tokenizer.

| | Qwen3-0.6B | Qwen3-1.7B |
|---|---|---|
| layers | 28 | 28 |
| KV heads | 8 | 8 |
| head_dim | 128 | 128 |
| hidden | 1024 | 2048 |
| rope_theta | 1e6 | 1e6 |
| tokenizer | Qwen2Tokenizer | identical |

The shared tokenizer is a requirement, not a convenience: the regression pairs KV
vectors *token by token*, so both models must see identical token streams.

Equal depth is a property **no pair in the paper has**. It removes the layer-alignment
confound and gives a clean identity baseline (layer *l* → layer *l*) to test whether
cross-layer top-k selection earns its complexity.

**Qwen3.5 0.8B/2B are out of scope** — they are hybrid linear-attention models
(`full_attention_interval: 4`, so only 6 of 24 layers keep a KV cache) and are 2 heads ×
256 dims, not matched-KV. See [REPORT.md §3](REPORT.md).

---

## Code structure

```
src/kvtransfer/
  rope.py           RoPE strip/apply in Qwen3's NeoX convention        [6 tests]
  capture.py        forward hooks on k_norm / v_proj → content space
  gram.py           streaming XtX/XtY accumulator (reference impl)     [8 tests]
  batched_gram.py   the same statistics as batched einsum grids        [6 tests]
  ridge.py          closed-form solve, unpenalized intercept, R²
  select.py         top-k cross-layer source selection + identity baseline
  mapper.py         fitted maps → live cache; zero-copy & mean controls
  graft.py          build a DynamicCache from content-space keys
  data.py           FineWeb-Edu calibration, WikiText-2 eval
  stream.py         chunked capture helpers
  eval/ppl.py       prefix-conditioned perplexity + oracle path

scripts/
  02_probe_r2.py       Stage A: the R² grid
  03_fit.py            Stage B: fit mappers at each k
  04_report_probe.py   turn a probe .pt into tables + heatmaps
  05_eval.py           perplexity vs all four baselines
  run_all.sh           the whole pipeline end to end

tests/
  test_rope.py         RoPE vs HF's own implementation
  test_ridge.py        streaming Gram == dense solve
  test_batched_gram.py batched grid == per-cell reference, to 1e-6
  test_identity.py     END-TO-END GUARDRAIL (marked `slow`)
```

### The guardrail

`tests/test_identity.py` fits a mapper from a model **to itself** and grafts the result
back. It must reproduce standalone logits. One test catches wrong RoPE convention, wrong
hook site, GQA head misalignment, position-id off-by-one, and cache layout
transposition. It runs first, because no downstream number means anything until it
passes.

Measured: **2.9e-05**.

---

## Setup

```bash
git clone <this repo> && cd kv-transfer
uv venv --python 3.12
uv pip install -e ".[dev]"
source .env.sh            # HF_HUB_DISABLE_XET=1 — see "Notes" below
```

Requires ~5 GB of disk for the two models and ~6 GB of free RAM.

## Running

```bash
# everything, in order, with the guardrails gating the rest
bash scripts/run_all.sh                    # 500 sequences (paper-faithful), CUDA
N_SEQS=100 bash scripts/run_all.sh         # quicker pass
DEVICE=cpu N_SEQS=50 bash scripts/run_all.sh
DEVICE=mps N_SEQS=50 bash scripts/run_all.sh   # Apple silicon
```

Or step by step:

```bash
pytest tests/ -q -m "not slow"     # 20 fast tests
pytest tests/ -q -m slow           # 4 end-to-end guardrails (downloads 0.6B)

python scripts/02_probe_r2.py --n-seqs 500 --device cuda --out results/probe.pt
python scripts/04_report_probe.py results/probe.pt
python scripts/03_fit.py --probe results/probe.pt --k 1 2 4 8 12 --identity-baseline
python scripts/05_eval.py --out results/eval_ppl.json
```

Configuration lives in `configs/qwen3_0.6b_to_1.7b.yaml` — model pair, calibration size,
chunk size, device, λ grid, k sweep. `configs/selftest_0.6b.yaml` runs the whole
pipeline 0.6B → 0.6B as a cheap smoke test.

### Expected runtime

| Hardware | 500 sequences | 100 sequences |
|---|---|---|
| A10 / L4 | ~30–40 min | ~10 min |
| M1 MPS | ~5 h | ~1 h |
| M1 CPU | ~19 h | ~4 h |

---

## Notes on small machines

Developed on an 8 GB M1, which shaped the design:

- **The design matrix is never materialized.** With ~128K observations per head and up
  to 3584 features it would be gigabytes; streaming `XᵀX`/`XᵀY` makes it megabytes. This
  is what makes the paper's closed form laptop-scale.
- **The two models are never resident at once.** Capture from one, free it, load the
  other.
- **Statistics are batched, not per-cell.** The obvious implementation allocates one
  accumulator per (src, tgt, head) cell — 12,768 small tensors totalling 1.7 GB, which
  fragments the heap badly enough to force swapping. `batched_gram.py` holds the same
  numbers in five contiguous tensors updated by one `einsum`, and is asserted equal to
  the per-cell version to 1e-6.
- **`xtx` is stored once per source layer**, not per cell: every target regresses on the
  same source features. A 28× saving on the largest array.
- **Gram accumulators are float32**, promoted to float64 inside `solve()` before the
  cancellation-prone centering step. Measured R² difference at this scale is under 1e-6.
- **`HF_HUB_DISABLE_XET=1`.** Hugging Face's Xet chunk store keeps a second copy of every
  downloaded blob; here it held 2.6 GB while the visible cache reported 4.3 MB.
- **No `shuffle(buffer_size=...)` on streaming datasets.** The buffer must fill before
  yielding anything: 12 minutes for a 4-sequence smoke test, versus 6.8 seconds without.

---

## Status

| Phase | State |
|---|---|
| 0 — verified math core | **done** — 20 fast tests + 4 guardrails green |
| 1 — calibration data | **done** |
| 2 — R² linear-structure probe | code done, self-tested; needs a machine that can hold the 1.7B |
| 3 — fit + k-sweep | code done |
| 4 — perplexity + retention | code done |
| 5 — writeup | pending results |

Everything is implemented and tested; the blocker is hardware. The development machine
runs at ~0.06 GB free with 9–12 GB of swap in use before this project starts, and
Qwen3-1.7B needs ~3.4 GB contiguous — loaded alone it is fine, but there is no room
alongside a normal desktop session. Details and measurements in
[REPORT.md §5](REPORT.md).
