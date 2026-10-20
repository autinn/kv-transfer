# Cross-Model KV Cache Transfer in the Qwen3 Family

A replication and extension of **"Cross-Model KV Cache Transfer in LLM Families: A
Closed-Form Linear Mapping for Prefill Reuse"** ([arXiv 2608.03893](https://arxiv.org/html/2608.03893v1),
Heo et al., NVIDIA). A small model reads the prompt; a ridge regression fitted in closed
form maps its KV cache into a larger model's format; the larger model answers without
running prefill itself.

Tested on four Qwen3 pairs, from 0.6B → 1.7B (below anything the paper reports) up to the
paper's own 14B → 32B, chosen so that depth mismatch and width mismatch can be separated.

---

## Results so far

HellaSwag (all 10,042 questions, zero-shot, acc_norm), retention = mapped / standalone
target. Mappers: all-heads features, λ = 0.01, 500 × 1,024 FineWeb-Edu calibration tokens.

| Pair | What changes | Zero-copy | k = 1 | **k = 8** | Held-out key R² (k = 8) |
|---|---|---:|---:|---:|---:|
| 0.6B → 1.7B | width (28 → 28 layers) | 56.1% | 89.3% | **92.2%** | 0.710 |
| 1.7B → 4B | depth + width (28 → 36) | 38.4% | 80.0% | **92.3%** | 0.707 |
| 4B → 8B | width (36 → 36) | 91.8% | 92.9% | **95.5%** | 0.752 |
| 14B → 32B | depth (40 → 64) | 68.4% | — | **97.8%** | 0.730 |

*Zero-copy* pastes the source cache in unmapped (layer-aligned): the baseline the
mapper has to beat. Every k = 8 vs zero-copy and k = 8 vs k = 1 difference is significant
on a paired test (`08_paired_test.py`; smallest p = 3e−21).

What the four pairs say so far:

1. **Retention tracks scale, not depth or width mismatch.** The pure-depth pair is the best.
2. **Depth mismatch changes the mechanism.** At equal depth one source layer does almost
   all the work (k = 8 adds ~2 points); at unequal depth multi-layer selection is the main
   ingredient (+8.4 points on 1.7B → 4B).
3. **Layer alignment is diagonal at equal depth and piecewise at unequal depth**, never
   the proportional stretch a naive alignment assumes.
4. **Qwen3-4B and 8B caches are nearly interchangeable unmapped** (92% zero-copy).

Sanity checks that gate every number: mapping a model onto itself reproduces its logits
(max error 2.9e−05); the oracle graft (target's own cache through the same graft path)
matches standalone within ±0.08 points; RoPE handling is checked against Hugging Face's.

### Reports, in order

| Report | What it covers |
|---|---|
| [REPORT.md](REPORT.md) | Pipeline and guardrails on an 8 GB M1; what was verified first |
| [REPORT2.md](REPORT2.md) | 0.6B → 1.7B: probe, mapper fits, perplexity; original proposal |
| [REPORT3.md](REPORT3.md) | 0.6B → 1.7B full run on an A10, the k ≥ 2 value-map bug fix, benchmark retention |
| [REPORT4.md](REPORT4.md) | Scoped replication of the paper's 14B → 32B on an H100 |
| [REPORT5.md](REPORT5.md) | 4B → 8B and 1.7B → 4B: the depth-vs-width matrix |
| [VIS.md](VIS.md) | Guide to the plots in `results/plots/` |
| [GPU.md](GPU.md), [NEXT_STEPS.md](NEXT_STEPS.md) | Rented-GPU setup and the run plan |

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
  stream.py         Gram accumulators for the top-k fit
  kvstore.py        capture KV once to disk shards; every later pass reads them
  runtime.py        model loading (device_map for big models), config helpers
  pipeline.py       config → calibration ids → source/target stores
  eval/ppl.py       prefix-conditioned perplexity + oracle path
  eval/harness.py   benchmark scoring from a transferred prefix cache
  eval/tasks.py     ARC-C, HellaSwag, WinoGrande, MMLU, GSM8K (lm-eval prompts)

scripts/
  02_probe_r2.py       Stage A: the R² grid
  03_fit.py            Stage B: fit mappers at each k
  04_report_probe.py   turn a probe .pt into tables + heatmaps
  05_eval.py           perplexity vs all four baselines
  07_eval_tasks.py     benchmark retention (the paper's headline metric)
  run_all.sh           the whole pipeline end to end (CONFIG=... for other pairs)

tests/
  test_rope.py         RoPE vs HF's own implementation
  test_ridge.py        streaming Gram == dense solve
  test_batched_gram.py batched grid == per-cell reference, to 1e-6
  test_identity.py     END-TO-END GUARDRAIL (marked `slow`)
  test_kvstore.py      store == direct capture; reuse, resume, invalidation
  test_unequal_depth.py  2-layer → 3-layer pair: alignment, zero-copy, validation
  test_harness.py      transferred-cache scoring == standalone scoring
  test_scripts_pipeline.py  every script end to end on a tiny unequal pair (`slow`)
```

### Pairs with different depths

Source and target may have different layer counts (Qwen3-14B has 40, Qwen3-32B has 64).
Wherever a "layer-aligned" source layer is needed (the zero-copy control, the identity
baseline, the RoPE comparison, the reports), target layer *l* reads source layer
`round(l · (Ls−1)/(Lt−1))`, which is *l* → *l* when depths match. Mappers record the
pair they were fitted on and refuse to run on another.

### Capture once

Each model's pre-RoPE KV for the calibration set is written to `runtime.store_dir` once
(`kvstore.py`). The probe, every k in the fit sweep, and the evaluations read those
shards; a stage whose store already exists never loads the model. A store is keyed by
model, token ids, stride and dtype, so a changed calibration set is recaptured rather
than silently reused, and an interrupted capture resumes at the missing shard.

### Benchmarks

`07_eval_tasks.py` scores each prompt with its context prefilled by the source, mapped,
and grafted into the target (`context[:-1]` is transferred; the target reads the last
context token and the answer). Standalone and oracle runs use identical scoring, and
`tests/test_harness.py` checks they agree. Big pairs run in disk-budgeted rounds
(`eval.store_budget_gb`) and resume from `eval_tasks.partial.json`. Tasks come from
`eval.tasks`; with none listed, only ARC-Challenge and WinoGrande (the smallest
downloads) run.

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
CONFIG=configs/qwen3_14b_to_32b.yaml bash scripts/run_all.sh   # paper's pair, 80 GB GPU
```

Or step by step:

```bash
pytest tests/ -q -m "not slow"     # 73 fast tests, no downloads
pytest tests/ -q -m slow           # guardrails on real 0.6B + tiny-pair script run

C=configs/qwen3_0.6b_to_1.7b.yaml
python scripts/02_probe_r2.py --config $C --device cuda      # -> <output_dir>/probe_r2_n500.pt
python scripts/04_report_probe.py results/probe_r2_n500.pt
python scripts/03_fit.py --config $C --k 1 2 4 8 12 --identity-baseline
python scripts/05_eval.py --config $C
python scripts/07_eval_tasks.py --config $C --limit 200     # benchmark retention
```

Configuration lives in `configs/` — model pair, `output_dir`, calibration size, shard
size, device, λ grid, k sweep, benchmark tasks. `qwen3_14b_to_32b.yaml` is the paper's
best pair (the direct check against its published numbers). `smoke_0.6b_to_1.7b.yaml`
runs every stage on the real small pair at laptop scale; `selftest_0.6b.yaml` runs it
0.6B → 0.6B.

> **Mappers fitted before this change with k ≥ 2 should be refit.** Keys and values are
> ranked separately in top-k selection and often pick different source layers, but the
> saved mapper recorded only the key selection, so value maps were applied to the wrong
> source layers at eval time. Loading such a mapper now warns.

### Feature width: same-head vs. all-heads

By default each target head is mapped from the *same-numbered* source head only
(`k·head_dim` inputs). The paper's method section states the feature vector instead
concatenates *all* source KV heads of each selected layer (`k·kv_heads·head_dim`
inputs), "enabling cross-head information flow" — matching its quoted mapper parameter
counts. Pass `--all-heads` to `03_fit.py` to fit that wider mapper (`stream.TopKGrams`
and `mapper.KVMapper` both take `all_heads=True`); it is not the default because it
costs more Gram memory and was added after REPORT3/REPORT4 both found the same
multi-layer R² shortfall at very different model scales, which points at this feature
difference rather than scale. See `NEXT_STEPS.md` for the comparison still to run.

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

## Reproducing the current results

**1. Quick check, any machine (~5 GB of downloads; minutes on a GPU, longer on CPU).** Runs every stage on the
real 0.6B → 1.7B pair at toy size. The numbers are not results; the check is that the
pipeline runs and the graft path is exact.

```bash
pytest tests/ -q -m "not slow"                       # expect: 73 passed
CONFIG=configs/smoke_0.6b_to_1.7b.yaml N_SEQS=16 DEVICE=cpu bash scripts/run_all.sh   # or DEVICE=mps / cuda
```

Expected at the end: a retention table where `standalone` and `oracle` are both 100.0 and
the mapper / zero-copy rows are below it (50 questions each, so they are noisy).

**2. One full pair, the protocol behind the results table (A100 80 GB, ≥120 GB disk).**

```bash
bash scripts/run_pair.sh configs/qwen3_4b_to_8b.yaml     # ~2 h
bash scripts/run_pair.sh configs/qwen3_1.7b_to_4b.yaml   # ~1.3 h
```

Expected output, in `results/<pair>/eval_tasks.json` (HellaSwag acc_norm):

| Config | 4B → 8B | 1.7B → 4B |
|---|---:|---:|
| standalone | 0.7494 | 0.6850 |
| oracle | 0.7501 | 0.6845 |
| zero_copy | 0.6879 | 0.2630 |
| mapper_k1_lam0.01_allheads | 0.6963 | 0.5482 |
| mapper_k8_lam0.01_allheads | **0.7156** | **0.6325** |

plus `heldout_r2.json` (k = 8 key R² ≈ 0.75 and 0.71) and `probe_report_n500.txt`.
14B → 32B: `configs/qwen3_14b_to_32b_replication.yaml`, see [REPORT4.md](REPORT4.md)
(k = 8 all-heads acc_norm 0.8093). The committed `results/` directories hold the outputs
of these runs; fitted mappers (2.3 GB each) are not committed.

---

## Status

| Part | State |
|---|---|
| Pipeline: capture → probe → fit → eval | **done**, guardrails green |
| Paper's 14B → 32B replication | **done**: 97.8% HellaSwag retention at k = 8 |
| Family matrix (four pairs) | **done**: REPORT5 |
| Other benchmarks, length generalisation, mapper anatomy | next |
| Write-up | in progress |
