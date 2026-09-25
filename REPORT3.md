# Experiment Report 3: Validation on a GPU Box, Bug Fix Confirmed

Replication of *Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear
Mapping for Prefill Reuse* (arXiv 2608.03893) on **Qwen3-0.6B → Qwen3-1.7B**.

**Date:** 2026-09-25
**Branch:** `cross-depth-benchmarks` (`12bc92a`) plus the uncommitted local changes in §2.2
**Hardware:** 1× NVIDIA A10 (23 GB), driver 570.148.08 (CUDA 12.8), 30 CPU cores,
222 GB RAM, 1.4 TB disk
**Software:** Python 3.12, PyTorch 2.11.0+cu128, transformers 5.17.0, datasets 5.0.1,
accelerate 1.15.0

Previous reports: [REPORT.md](REPORT.md) (implementation and verification, M1) and
[REPORT2.md](REPORT2.md) (first GPU run, 100 calibration sequences).

---

## 1. Summary

- **The value-map bug fix works, and it changes the conclusion of REPORT2.** REPORT2
  found that held-out perplexity got *worse* past k=8 (k=16: 344) and read that as
  overfitting. With the fix, perplexity improves steadily with k. The best mapper goes
  from **29.04 to 20.26** perplexity (standalone 16.70, zero-copy 59.84), i.e.
  log-perplexity retention from ~80% to **93.6%**. The k=16 collapse was the bug, not
  overfitting.
- **All guardrails pass on this hardware.** The oracle graft reproduces standalone
  perplexity (16.699 vs 16.698) and standalone benchmark accuracy (identical on ARC-C,
  within fp16 tie-flips on HellaSwag and WinoGrande).
- **Calibration is now paper-faithful (500 sequences).** Fit R² is slightly *lower*
  than REPORT2's. That's expected: REPORT2 used 100 sequences, and in-sample R² is
  optimistic at small n.
- **λ = 0.001 beats the paper's 0.01 on held-out perplexity at every k ≥ 2** (λ = 0.1
  is worst). This is consistent with our λ being a variance-scaled penalty rather than
  the paper's raw λ (§7).
- **HellaSwag accuracy retention: 81.2% at k=1 and 89.0% at k=8** (λ=0.01), against
  55.8% for zero-copy and 99.9% for the oracle. The mappers recover 57% (k=1) and 75%
  (k=8) of the gap between zero-copy and standalone. The paper reports 97.6% average
  retention at k=8 on 14B → 32B. To save GPU time, HellaSwag was the only benchmark run
  on mappers (§5.4, §8).
- **Five issues were found and fixed or worked around locally**, including one bug that
  would have crashed the 14B → 32B probe (§2.2).
- **Against the paper (§6): qualitatively a positive replication at small scale, quantitatively weaker.**
  Keys beat values, RoPE stripping helps, and more source layers help, but fit quality is
  lower (best key R² 0.62 vs 0.81; k=8 0.65 vs 0.79).
- **Not run: Qwen3-14B → 32B.** It needs ~80 GB of GPU memory, and this box has 23 GB.

---

## 2. What changed since REPORT2

### 2.1 Code on the branch (`12bc92a`)

| Change | Effect on this experiment |
|---|---|
| **Value-map fix.** Keys and values are ranked separately in top-k selection and often pick different source layers (17/28 target layers differ at k=2, 22–27/28 at k ≥ 8 in this run). Saved mappers used to record only the key selection, so value maps ran on the wrong source layers at eval time. Mappers now store both. | Every k ≥ 2 number in REPORT2 was measured on partly miswired mappers. This run is the first clean measurement. |
| **Capture once to disk** (`kvstore.py`). Each model's pre-RoPE KV is written to shards once, keyed by a fingerprint of model, token ids, stride and store dtype. | Probe, fit and evals read the shards. Verified: the fit printed "reusing KV store" and loaded no model. |
| **Benchmark eval** (`07_eval_tasks.py`). ARC-C, HellaSwag, WinoGrande, MMLU and GSM8K, scored from a transferred prefix cache, in disk-budgeted, resumable rounds. | First downstream-accuracy measurement for this pair (REPORT2 listed it as unmeasured). |
| **Unequal-depth support** for 14B (40 layers) → 32B (64 layers). | Reduces to l → l on this equal-depth pair. Covered only by the tiny 2 → 3 layer tests here. |

### 2.2 Local changes made during this validation (uncommitted)

| # | Issue | Fix | Where |
|---|---|---|---|
| 1 | `uv` not installed | installed via the official installer | GPU.md §1 |
| 2 | The default torch wheel (2.14+cu130) needs a CUDA 13 driver. On this CUDA 12.8 driver `torch.cuda.is_available()` was silently `False`. | reinstalled torch 2.11.0+cu128 | GPU.md §2 |
| 3 | **`accelerate` is missing from `pyproject.toml`.** Every pipeline script loads with `device_map` and crashed at model load. The tests load on CPU without it, so they didn't catch it. | `uv pip install accelerate`. Repo fix: add it to `dependencies`. | GPU.md §3 |
| 4 | **`torch.eye` created without a device** in `BatchedGrid.solve_r2` / `DiagonalGrid.solve_r2`. With `probe.gram_device: cuda` the probe crashed at the solve step. **`configs/qwen3_14b_to_32b.yaml` sets exactly that option**, so the paper-pair run would have crashed the same way. | `device=cxx.device` (2 lines). Fast tests pass, and GPU R² is identical to CPU R². | `src/kvtransfer/batched_gram.py:84,145` |
| 5 | The small-pair configs were sized for an 8 GB M1 (mps, batch 1, CPU Grams, no benchmark tasks). | A10 configs: `configs/qwen3_0.6b_to_1.7b.a10.yaml` and `configs/smoke_0.6b_to_1.7b.a10.yaml`. Memory limits measured on the A10. | GPU.md §4 |

Also found while doing this (not fixed, documented):

- **`calibration.dataset` is ignored** except for `"synthetic"`. Any other value
  silently calibrates on FineWeb-Edu, and the calibration token cache
  (`results/calib_{n}x{len}_s{seed}.pt`) isn't keyed by dataset. A calibration-domain
  ablation needs code changes, not just a config change.
- **The fit accumulates Grams in fp64 with no config option.** The A10 runs fp64 at
  ~1/32 of its fp32 rate, so fit time grows steeply with k (k=1 0.3 min, k=8 6.5 min,
  k=16 21.8 min).
- **The identity test's tolerance is `atol=2e-3`**, looser than the ~1e-5 it's usually
  described as. The measured max logit error is 1.5e-4 (fp32, CPU), about 1e-5 relative
  to logit magnitude.

### 2.3 Settings compared with REPORT2

| | REPORT2 | This run |
|---|---|---|
| Calibration | FineWeb-Edu, **100** seqs × 1,024, stride 4 (25.6K obs/head) | FineWeb-Edu, **500** seqs × 1,024, stride 4 (128K obs/head, the paper's setting) |
| Mapper code | before the value-map fix | after the fix |
| k / λ sweep | {1,2,4,8,12,16} / {0.001, 0.01, 0.1} | same |
| Perplexity eval | WikiText-2, 64 seqs, 512-token prefix | same |
| Benchmarks | none | ARC-C, HellaSwag, WinoGrande (+ MMLU, GSM8K, see §8) |
| Compute | GPU, `batch_size: 1`, CPU Grams | A10, batch 8, GPU Grams |

---

## 3. What is being tested

1. **Environment and guardrails on real CUDA hardware.** Do the fast tests, the slow
   real-model tests and the identity graft still pass?
2. **The real Qwen3 code path at smoke scale** (16 seqs) before spending GPU time: store
   reuse, oracle = standalone, oracle benchmark retention ≈ 100%.
3. **The value-map bug fix.** Do multi-layer mappers now improve held-out perplexity as
   k grows, instead of collapsing?
4. **Paper-faithful calibration size** (500 seqs), and how it changes fit and
   perplexity relative to REPORT2's 100.
5. **Downstream accuracy retention**, the paper's headline metric, measured for the
   first time on this pair.

---

## 4. Validation steps (Steps 0–3)

| Step | Result |
|---|---|
| 0. Branch | `cross-depth-benchmarks` at `12bc92a` |
| 1. Environment | A10 with 23 GB (22.6 GB free), 1.3 TB disk free. Needed the setup steps in §2.2 #1–3. |
| 2a. Fast tests | **57 passed** (6 s) |
| 2b. Slow tests (real Qwen3-0.6B, tiny 2 → 3 layer pipeline) | **8 passed** (66 s). Identity graft max logit error 1.5e-4. |
| 3. Smoke run, 0.6B → 1.7B, 16 seqs | **All three checks pass.** Fit reused both stores and loaded no model. Oracle perplexity 18.154 vs standalone 18.153. Oracle retention 100% on ARC-C and WinoGrande. Mapper accuracies at 50 questions per task are noise (all near chance), as expected. |

The smoke run also exposed issue #4 (GPU Grams), which was fixed and re-verified: GPU R²
is identical to the CPU path to 3 decimals.

---

## 5. Results: full run, 500 calibration sequences

Stage timings on the A10: guardrails ~1.5 min, KV capture ~1 min per model, probe
2.8 min total, fit 44 min total (dominated by fp64 at k=12 and k=16), perplexity eval ~3 min.

### 5.1 Linear-structure probe

| Quantity | Best R² | Layer-aligned mean | REPORT2 (n=100) |
|---|---:|---:|---|
| Content-space keys | 0.620 | 0.532 | 0.625 / 0.537 |
| Post-RoPE keys | 0.613 | 0.518 | 0.618 / 0.523 |
| Values | 0.503 | 0.401 | 0.509 / 0.408 |

- The best source layer is the aligned one for **28/28** target layers, as in REPORT2.
- Stripping RoPE helps (0.518 → 0.532 aligned mean), as in REPORT2.
- Keys are more predictable than values, matching the paper's ordering.
- The best single-source key R² is 0.620, below the paper's 0.81 for 14B → 32B.
- All values are ~0.005 lower than at n=100: less in-sample optimism with 5× the data.

### 5.2 Mapper fit (mean in-sample R², keys / values)

| k | λ=0.001 | λ=0.01 | λ=0.1 | REPORT2 λ=0.001 (n=100, pre-fix) | K/V source layers differ |
|---|---|---|---|---|---|
| 1 | 0.536 / 0.401 | 0.532 / 0.401 | 0.507 / 0.397 | 0.541 / 0.408 | n/a |
| 2 | 0.579 / 0.445 | 0.574 / 0.444 | 0.550 / 0.426 | 0.586 / 0.455 | 17/28 |
| 4 | 0.619 / 0.489 | 0.614 / 0.480 | 0.590 / 0.447 | 0.630 / 0.505 | 25/28 |
| 8 | 0.655 / 0.533 | 0.650 / 0.520 | 0.626 / 0.486 | 0.672 / 0.557 | 22/28 |
| 12 | 0.672 / 0.552 | 0.667 / 0.538 | 0.643 / 0.506 | 0.694 / 0.584 | 27/28 |
| 16 | 0.683 / 0.568 | 0.678 / 0.552 | 0.654 / 0.517 | 0.711 / 0.606 | 27/28 |
| identity | = k=1 | = k=1 | = k=1 | = k=1 | |

**The gap to REPORT2 grows with k** (−0.005 at k=1, −0.028 at k=16 for keys). This
matches in-sample R² inflation of roughly (1−R²)·features/observations: k·128 features
against 25.6K observations at n=100 vs 128K at n=500. The new numbers are the more
honest ones. The bug fix doesn't show up in fit R² at all, because R² was always
computed on the correct features and the bug only affected eval.

**Comparison with the paper (different, larger pair):** paper k=1 0.56 / 0.32 and k=8
0.79 / 0.65. Ours: k=1 0.532 / 0.401, k=8 0.650 / 0.520. Keys at k=1 are close. The
gain from multiple layers is smaller here (0.53 → 0.65 vs the paper's 0.56 → 0.79).

### 5.3 Held-out perplexity (WikiText-2, 64 seqs, 512-token prefix)

| Configuration | This run λ=0.001 | This run λ=0.01 | This run λ=0.1 | REPORT2 λ=0.01 (pre-fix) |
|---|---:|---:|---:|---:|
| Target standalone | 16.698 | | | 16.699 |
| Oracle graft | 16.699 | | | 16.700 |
| k=1 (= identity) | 31.69 | 31.98 | 38.81 | 31.92 |
| k=2 | 29.53 | 30.57 | 37.68 | 31.98 |
| k=4 | 24.48 | 25.63 | 32.81 | 41.30 |
| k=8 | 21.57 | 22.48 | 27.34 | 29.04 |
| k=12 | 20.74 | 21.56 | 25.03 | 111.13 |
| k=16 | **20.26** | 21.00 | 24.01 | 344.05 |
| Zero-copy | 59.84 | | | 59.85 |
| Mean prediction | 655.0 | | | 628.1 |

Log-perplexity retention (`log ppl_standalone / log ppl`): best **93.6%** (k=16,
λ=0.001), k=8 at λ=0.01 90.4%, k=1 81.2%, zero-copy 68.8%.

**Separating the bug fix from the extra data:** k=1 uses one source layer for keys and
one for values, so the bug couldn't affect it. Going from 100 to 500 sequences moved k=1
by only 0.06 perplexity (31.92 → 31.98). The large changes at k ≥ 4 are therefore
mostly the bug fix. More data probably also helps at high k, where features per
observation are highest, but that share isn't isolated (a fixed-code n=100 refit would
measure it; not run).

**λ:** less regularization generalizes better at every k ≥ 2. The paper's λ=0.01 isn't
the best setting under this repo's scaling (§7).

### 5.4 Benchmark retention

Zero-shot, lm-evaluation-harness prompt formats, full test sets. Mapper
configurations use λ=0.01 (the paper's value). To limit GPU time, **only HellaSwag was
run on the mappers**, and only at k=1 and k=8, the two points the paper reports for it.
ARC-C and WinoGrande have the baselines only (§8).

**HellaSwag (10,042 questions, acc_norm):**

| Configuration | acc_norm | Retention vs standalone | Share of the zero-copy → standalone gap recovered |
|---|---:|---:|---:|
| Standalone (1.7B) | 0.6035 | 100% | — |
| Oracle graft | 0.6030 | 99.9% | — |
| **k=8, λ=0.01** | **0.5369** | **89.0%** | **75.0%** |
| **k=1, λ=0.01** | **0.4899** | **81.2%** | **57.4%** |
| Zero-copy | 0.3368 | 55.8% | 0% |
| Chance | 0.25 | | |

- **The mapping carries real information.** k=1 is 15.3 points above zero-copy and k=8 is
  20.0 points above. The standard error of each accuracy is ~0.5 points, so both gaps
  are far outside noise.
- **More source layers help on the benchmark too:** k=8 beats k=1 by 4.7 points
  (+7.8 retention points), well outside noise.
- **Retention is a ratio to standalone**, so it counts accuracy the model gets even at
  chance. The "gap recovered" column removes that floor: k=8 recovers three quarters of
  what zero-copy loses.
- Runtime: ~10–11 min per mapper configuration on the A10 (k=1 594 s, k=8 669 s).

**Baselines on all three log-likelihood tasks:**

| Configuration | ARC-C acc_norm | HellaSwag acc_norm | WinoGrande acc |
|---|---:|---:|---:|
| Standalone (1.7B) | 0.4317 | 0.6035 | 0.6077 |
| Oracle graft | 0.4317 (100.0%) | 0.6030 (99.9%) | 0.6085 (100.1%) |
| Zero-copy | 0.2756 (63.8%) | 0.3368 (55.8%) | 0.5075 (83.5%) |
| Chance | 0.25 | 0.25 | 0.50 |

The oracle rows are the harness self-check: ARC-C is identical, HellaSwag differs on
5 of 10,042 questions, and WinoGrande on 1 of 1,267. These are fp16 near-tie flips, not a
systematic error. Zero-copy is at or near chance on all three tasks, as expected for an unmapped cache.
Any mapper accuracy above these rows is information the mapping carries over.
WinoGrande's zero-copy retention (83.5%) looks high only because chance there is 50%.

---

## 6. How the replication compares with the paper

The paper's numbers are for **Qwen3-14B → 32B**. Ours are for **0.6B → 1.7B**, a pair
~20× smaller than anything the paper reports. So this is a sanity comparison of the
method's behavior at small scale, not a replication of the paper's numbers. The direct
check (14B → 32B) still needs an 80 GB GPU.

### 6.1 Side by side

| Measure | Paper (14B → 32B) | Ours (0.6B → 1.7B) | Read |
|---|---|---|---|
| Best single-source key R² | 0.81 | 0.620 | Lower, as expected for a smaller pair |
| k=1 mean R², keys / values | 0.56 / 0.32 | 0.532 / 0.401 | Close; our values are higher |
| k=8 mean R², keys / values | 0.79 / 0.65 | 0.650 / 0.520 | Multi-layer gain is smaller: +0.12 keys vs the paper's +0.23 |
| Keys more predictable than values | yes | yes (every k) | Reproduces |
| Stripping RoPE helps the key fit | yes | yes, 0.518 → 0.532 aligned mean | Reproduces (small gain; both models share `rope_theta`) |
| More source layers → better transfer | yes (k=1 → k=8) | yes: perplexity 31.98 → 22.48 (λ=0.01), 20.26 at k=16, λ=0.001 | Reproduces, but only after the value-map fix |
| Harness: oracle = standalone | implied | perplexity 16.699 vs 16.698; accuracy 100.0 / 99.9 / 100.1% | Verified |
| Benchmark retention at k=8 | **97.6%** average over 5 tasks | **89.0%**, HellaSwag only | Lower; §6.3 |
| HellaSwag, k=1 → k=8 | 44.81 → 80.70 (+35.9 pts) | 48.99 → 53.69 (+4.7 pts); standalone 60.35 | Direction reproduces; gain much smaller, starting from a stronger k=1 (§6.3) |
| λ | 0.01 | 0.001 best on held-out perplexity at every k ≥ 2 | Not comparable as-is: our λ is variance-scaled (§7) |

### 6.2 What reproduces

- **The linear structure exists at sub-2B scale.** Single-layer maps explain about half
  the key variance (0.53) and 40% of the value variance. That's weaker than the
  paper's best pair but far from zero. REPORT.md set R² ≥ 0.4 as "partial structure,
  clearly non-trivial"; this result is in that band.
- **The paper's qualitative findings reproduce:** keys beat values, RoPE stripping helps,
  and multiple source layers beat one.
- **Transfer is real:** the best mapper gets held-out perplexity to 20.26 against 16.70
  standalone and 59.84 for zero-copy. On HellaSwag, the k=8 mapper reaches 89.0%
  retention and recovers 75% of the gap between zero-copy and standalone.
- **More source layers help downstream, not just in R²:** HellaSwag improves from k=1
  to k=8 by 4.7 points, as in the paper.

### 6.3 Where it falls short, and possible reasons

- **Fit quality is lower** (0.62 vs 0.81 best key R²; 0.65 vs 0.79 at k=8). Candidate
  explanations, none isolated yet:
  1. **Scale.** Small models have had less chance to converge on shared
     representations. This is the question the repo was built to answer (README).
  2. **Features per target head.** This repo maps each target head from the same-numbered
     source head only (k·128 inputs). The paper's parameter counts suggest reading all 8
     source heads (k·8·128), which would add capacity, especially at larger k, where our
     gain is weaker than the paper's.
  3. **λ convention.** Our variance-scaled λ=0.01 is likely a much stronger penalty than
     the paper's raw 0.01. λ=0.001 already beats it on held-out perplexity.
- **Benchmark retention is lower:** 89.0% on HellaSwag at k=8, against the paper's
  97.6% five-task average for 14B → 32B. That's a single task against an average, so the
  gap is indicative, not exact.
- **The k=1 → k=8 pattern differs from the paper's.** The paper goes 44.81 → 80.70 on
  HellaSwag, so its k=1 is weak: at most ~56% retention if its standalone accuracy is at
  least the k=8 score. It jumps by 36 points with more layers. Ours starts much stronger
  (81.2% at k=1) and gains only 4.7 points. The fit shows the same shape: our k=1 value
  R² is above the paper's (0.40 vs 0.32), but our k=1 → k=8 gain is about half the
  paper's (+0.12 vs +0.23 keys). So on this equal-depth pair, the aligned single layer
  already carries most of what the linear map can transfer, and extra layers add
  little. Reason 2 above (same-numbered source head only) would limit exactly this
  multi-layer gain. That makes it the most useful thing to test next.

### 6.4 Verdict

**Qualitatively a positive replication at small scale; quantitatively weaker.** The
method works on 0.6B → 1.7B, and every qualitative claim checked reproduces:
- linear structure exists (key R² ≈ 0.53 at k=1)
- keys are more predictable than values
- stripping RoPE helps
- more source layers help, in both perplexity and HellaSwag accuracy
- the mapped cache carries real information (HellaSwag 89.0% retention at k=8, 75% of
  the zero-copy gap recovered)

Transfer is less complete than in the paper's 14B → 32B results (89.0% vs 97.6%), and
the multi-layer gain is much smaller. Whether that gap comes from scale, head-feature
layout or λ can't be separated on this pair alone. The feature-layout and λ questions
can be tested cheaply here; scale needs the 14B → 32B run.

---

## 7. Differences from the paper's protocol

Found by checking the paper (§3.1, §4, Appendices C–D) against the repo.

| Item | Paper | This repo / run | Status |
|---|---|---|---|
| Model pair | Qwen3-14B → 32B (and others ≥ 3B) | 0.6B → 1.7B | by design; 14B → 32B needs an 80 GB GPU |
| MMLU | 5-shot | 0-shot | mismatch |
| GSM8K | 8-shot chain-of-thought | 5-shot, random train examples | mismatch |
| WikiText-2 perplexity | 1,024-token prefix, 2,048-token chunks | 512-token prefix, 1,024-token sequences | mismatch (cheap to rerun) |
| λ | 0.01 (convention not verified) | 0.01 × mean feature variance | open; our λ=0.001 wins on held-out perplexity |
| Features per target head | parameter counts suggest all 8 source heads | same-numbered head only | open (NEXT_STEPS.md) |
| Table 9 benchmarks: ARC-Easy, PIQA, BoolQ | used | not implemented | out of scope for this validation |
| Calibration-domain ablation (Wikipedia, CodeAlpaca) | Tables 8–9, on 14B → 32B | `calibration.dataset` ignored (§2.2) | out of scope |
| CoQA multi-turn switching | §4.6, 14B ↔ 32B | no harness | out of scope |

Retention is a ratio of two runs under the same prompts, so the shot-count mismatches
matter less for retention than for absolute accuracy. They still prevent a
protocol-matched comparison on MMLU and GSM8K.

---

## 8. Run status: stopped here

The run is **complete for this validation**. The GPU is idle.

**What was run:**

| Stage | Status |
|---|---|
| Guardrails (fast + slow tests) | done |
| Smoke run, 16 seqs | done |
| Probe, n=500 | done |
| Fit, k ∈ {1,2,4,8,12,16} + identity × λ ∈ {0.001, 0.01, 0.1} | done |
| WikiText-2 perplexity, all 21 mappers + 4 baselines | done |
| Benchmarks: standalone, oracle, zero-copy on ARC-C, HellaSwag, WinoGrande | done |
| Benchmarks: k=1 and k=8 (λ=0.01) on HellaSwag | done |

**What was stopped or skipped to save GPU time:**
- The original benchmark plan was 5 tasks × 9 configurations, estimated at ~15 h, most
  of it GSM8K generation. It was stopped during round 1.
- **MMLU and GSM8K** were skipped. Both are also protocol-mismatched with the paper
  (§7).
- **ARC-C and WinoGrande** have baselines only, no mapper rows.
- **k=4, k=16 and identity** were not run on benchmarks (identity equals k=1 on this
  pair). Only ARC-C has a mean-prediction score (0.2526); it was skipped elsewhere.
- **Not planned:** 14B → 32B (hardware), the Table 9 benchmarks and calibration-domain
  ablation, CoQA.

**If more GPU time becomes available,** in order of value per GPU minute:
1. ARC-C on k=1 and k=8: ~5 min, including a ~1 min recapture of source KV.
2. HellaSwag on k=16 (the best perplexity mapper): ~12 min.
3. A cheap test of §6.3's reason 2 or 3: a λ=0.001 HellaSwag row (~12 min, no refit
   needed), or an all-heads feature mode (needs code).
4. The 4,096-token length check or the paper's perplexity protocol: ~10 min each.

The run resumes from `results/eval_tasks.json`, so none of these repeat finished work.
Add configurations with `--only` and tasks with `--tasks`.

**Local state:** nothing is committed. The changes are the `batched_gram.py` fix, the
`*.a10.yaml` configs, GPU.md, REPORT3.md and `logs/`. Benchmark KV stores were deleted.
The calibration and perplexity stores (~32 GB) remain in `results/kv/`.

---

## 9. Reproducing

Setup: [GPU.md](GPU.md) §1–3. Then:

```bash
source .venv/bin/activate && source .env.sh && export TOKENIZERS_PARALLELISM=false
pytest tests/ -q -m "not slow" && pytest tests/ -q -m slow

C=configs/smoke_0.6b_to_1.7b.a10.yaml            # Step 3 smoke run
python scripts/02_probe_r2.py --config $C --device cuda
python scripts/03_fit.py --config $C --device cuda --identity-baseline
python scripts/05_eval.py --config $C --device cuda
python scripts/07_eval_tasks.py --config $C --device cuda

CONFIG=configs/qwen3_0.6b_to_1.7b.a10.yaml N_SEQS=500 KS="1 2 4 8 12 16" \
  bash scripts/run_all.sh                         # full run
```

Logs: `logs/smoke_*.log`, `logs/small_full.log` (probe → perplexity),
`logs/small_tasks5.log` (stopped five-task benchmark), `logs/hellaswag.log`
(HellaSwag mapper runs), `logs/old_run/` (REPORT2's `eval_ppl.json` and
fit plots). Results: `results/probe_r2_n500.pt`, `results/probe_report_n500.txt`,
`results/mappers/`, `results/eval_ppl.json`, `results/eval_tasks{,.partial}.json`.
