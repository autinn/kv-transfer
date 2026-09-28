# Experiment Report 4: Scoped Replication on the Paper's Pair (Qwen3-14B → 32B)

Replication check of *Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear
Mapping for Prefill Reuse* (arXiv 2608.03893) on the paper's own best pair,
**Qwen3-14B → Qwen3-32B**.

**Date:** 2026-09-28
**Branch:** `cross-depth-benchmarks`. Run on `123a5ee`; the fixes found during it are committed in `637349b`, `a24dd5c` and `a171632` (§5)
**Hardware:** 1× NVIDIA H100 PCIe (80 GB), driver 570.148.08 (CUDA 12.8), 26 CPU cores,
221 GB RAM, 993 GB disk
**Software:** Python 3.12, PyTorch 2.11.0+cu128, transformers 5.17.0, accelerate 1.15.0
**Config:** `configs/qwen3_14b_to_32b_replication.yaml` (k=8, λ=0.01, HellaSwag only)

Previous report: [REPORT3.md](REPORT3.md) (0.6B → 1.7B on an A10).

---

## 1. Summary

| Measure | Paper (14B → 32B) | This run (14B → 32B) | REPORT3 (0.6B → 1.7B) |
|---|---:|---:|---:|
| Best single-source key R² | 0.81 | **0.771** | 0.620 |
| k=1 mean R², keys / values | 0.56 / 0.32 | **0.558 / 0.455** | 0.532 / 0.401 |
| k=8 mean R², keys / values | 0.79 / 0.65 | **0.669 / 0.545** | 0.650 / 0.520 |
| HellaSwag retention at k=8 | 97.6% (5-task avg) | **95.8%** (HellaSwag only) | 89.0% |
| HellaSwag acc_norm at k=8 (absolute) | 80.70 | **79.28** | 53.69 |
| WikiText-2 log-ppl retention at k=8 | — | **92.4%** (ppl 8.98 vs 7.61) | 90.4% (ppl 22.48 vs 16.70) |

- **Single-layer structure reproduces.** Best key R² is 0.771 against the paper's 0.81, and
  the k=1 key mean (0.558) matches the paper's 0.56 almost exactly.
- **The multi-layer gain does not.** k=1 → k=8 raises key R² by +0.11 here, against the
  paper's +0.23. This is the same shortfall REPORT3 saw on the small pair (+0.12), so
  scale is not what limits it. That points at REPORT3's reasons 2 and 3
  (same-numbered source head only; variance-scaled λ) rather than model size.
- **Downstream transfer nearly reproduces.** The k=8 mapper keeps **95.8%** of standalone
  HellaSwag accuracy (79.28 vs 82.72 acc_norm) and recovers **86.8%** of the gap between
  zero-copy and standalone. Its absolute score is within 1.4 points of the paper's 80.70
  for the same configuration. The paper's 97.6% is a five-task average, so it isn't
  directly comparable with this HellaSwag-only figure.
- **The harness self-check passes.** The oracle graft scores 8,299 correct against
  standalone's 8,307 out of 10,042 (99.9% retention).
- **Verdict: a quantitative replication at the paper's scale on single-layer structure and
  on HellaSwag retention, with a weaker multi-layer R² gain than reported.** Scale
  explains most of REPORT3's gap to the paper on retention (89.0% → 95.8%), but not the
  multi-layer gain.

---

## 2. Setup and guardrails

| Step | Result |
|---|---|
| 0. Environment | GPU.md §1–3 applied as written (same driver as the A10). `accelerate` is still missing from `pyproject.toml`; installed by hand. `torch.cuda.is_available()` = True. The `torch.eye` device fix from REPORT3 is already committed. |
| 1a. Fast tests | **57 passed** |
| 1b. Slow tests (real Qwen3-0.6B) | **8 passed** (118 s) |

Weights were downloaded in parallel with the start of the probe (~100 MB/s combined), so the
14B capture started as soon as its weights landed.

---

## 3. Stage timings (H100)

| Stage | Time | Notes |
|---|---|---|
| KV capture, 14B, 500 × 1,024 | ~3 min | 21.0 GB store |
| KV capture, 32B, 500 × 1,024 | ~3.5 min | 33.6 GB store; started when the 32B download finished |
| Probe (whole script, incl. captures and download wait) | 12.6 min | |
| Fit, k=8, λ=0.01 | **~35 s** | vs 6.5 min for k=8 on the A10's small pair: the H100's fp64 rate removes the fit bottleneck |
| HellaSwag benchmark, 4 configs | ~1 h 35 min wall time | 2 rounds under `store_budget_gb: 60`. Includes two crashes and one 8.7 min recapture (§5); clean scoring time was ~66 min |

---

## 4. Results

### 4.1 Linear-structure probe (n = 500, 128K observations per head)

| Quantity | Best R² | Mean of best source per target | Layer-aligned mean |
|---|---:|---:|---:|
| Content-space keys | **0.771** | **0.558** | 0.403 |
| Post-RoPE keys | 0.749 | — | 0.340 |
| Values | 0.737 | 0.455 | 0.185 |

- **Stripping RoPE helps** (aligned mean 0.340 → 0.403), as in the paper and REPORT3.
- **Keys beat values**, as in the paper.
- **The "layer-aligned" column is misleading for unequal depths.** It maps target layer t
  to source ⌊t·40/64⌋, but that is the best source layer in only 8/64 target layers.
  The real best-source path is:
  - target 0–31 ← source 0–31 (same index)
  - target 32–60 ← source 8–36 (target − 24)

  The second half of the 32B lines up with the 14B's layers from 8 onward, as if the
  extra 24 layers repeated a block of the 14B's middle rather than stretching it. The
  "mean of best source per target" column is the fair k=1 comparison with the paper.
- `04_report_probe.py` ends with a hard-coded line, "The method appears to hold at sub-2B
  scale", which is wrong for this pair; it is left over from the small-pair runs.

### 4.2 Mapper fit (k=8, λ=0.01)

| | Keys | Values |
|---|---:|---:|
| Mean in-sample R², this run | **0.669** | **0.545** |
| Paper | 0.79 | 0.65 |
| REPORT3 (0.6B → 1.7B) | 0.650 | 0.520 |

- K and V selected different source layers in **63/64** target layers, so the value-map
  fix from REPORT3 is active (saved mappers carry separate K and V selections).
- The whole k=8 fit ran as one target-layer group under `max_gram_gb: 40`.

### 4.3 HellaSwag retention (10,042 questions, acc_norm, zero-shot)

| Configuration | acc_norm | Correct / 10,042 | Retention vs standalone | Share of zero-copy → standalone gap recovered |
|---|---:|---:|---:|---:|
| Standalone (32B) | 0.8272 | 8,307 | 100% | — |
| Oracle graft | 0.8264 | 8,299 | 99.9% | — |
| **k=8, λ=0.01** | **0.7928** | **7,961** | **95.8%** | **86.8%** |
| Zero-copy | 0.5661 | 5,685 | 68.4% | 0% |
| Chance | 0.25 | | | |

(acc, unnormalized: standalone 0.6384, oracle 0.6389, k=8 0.6150, zero-copy 0.4447.)

- **Harness self-check:** oracle and standalone differ by 8 net questions out of 10,042.
  Per-context, the raw log-likelihood argmax differs in 51 of 10,019 contexts, and the
  median |Δ log-likelihood| is 0.16 nats. That is noisier than REPORT3's fp16 1.7B run
  (5 flips), as expected for bf16 at 32B through two different compute paths
  (full-sequence forward vs grafted bf16 cache). It's not a systematic error: the net
  effect is 0.1 points.
- **Against the paper:** the paper reports 80.70 acc_norm on HellaSwag at k=8 for this
  pair; this run gets 79.28. The paper's 97.6% retention is a five-task average; this run's
  95.8% is HellaSwag alone.
- **Against REPORT3's small pair:** retention rises from 89.0% to 95.8%, and the share of
  the zero-copy gap recovered from 75.0% to 86.8%. Fit R² barely changed between the pairs
  (k=8 keys 0.650 → 0.669), so in-sample R² understates how much better the larger pair
  transfers.
- Zero-copy (68.4%) retains more here than on the small pair (55.8%): unmapped 14B
  content-space KV already carries some usable signal for the 32B.

### 4.4 Held-out perplexity (WikiText-2, 64 seqs, 512-token prefix)

Run after the benchmark at the user's request (outside the original scope); same protocol
as REPORT3 §5.3. Retention is `log ppl_standalone / log ppl`.

| Configuration | ppl | Retention | REPORT3 (0.6B → 1.7B) |
|---|---:|---:|---:|
| Target standalone (32B) | 7.608 | 100.0% | 16.698 / 100.0% |
| Oracle graft | 7.603 | 100.0% | 16.699 / 100.0% |
| **k=8, λ=0.01** | **8.983** | **92.4%** | 22.482 / 90.4% |
| Zero-copy | 46.611 | 52.8% | 59.836 / 68.8% |
| Mean prediction | 138.072 | 41.2% | 655.0 / 43.4% |

- The oracle matches standalone (7.603 vs 7.608), so the graft path is exact here too.
- k=8 retention rises only from 90.4% to 92.4% between the pairs, much less than
  HellaSwag's 89.0% → 95.8%. The mapped 14B cache costs 1.4 ppl points on the 32B.
- Zero-copy drops from 68.8% to 52.8%. This is the opposite of HellaSwag, where zero-copy
  retained more on the big pair. Perplexity punishes every token of a wrong cache, while a
  4-way multiple choice only needs the right ranking.

---

## 5. Eval batching change and fixes

During the benchmark's source-capture phase GPU utilization was 26–39%. Two code paths ran
at effective batch 1 on HellaSwag:

1. **`build_store` (`src/kvtransfer/kvstore.py`)** batched only consecutive
   *equal-length* items, and HellaSwag contexts all differ in length. Now items are sorted
   by length, right-padded into batches of `runtime.batch_size`, and each item's KV is
   sliced back to its own length. Causal attention means padding after an item cannot
   change its KV.
2. **`score_contexts` (`src/kvtransfer/eval/harness.py`)** ran one forward per context.
   New `score_batch` puts several contexts in one forward, up to `eval.score_batch_tokens`
   (default 8,192) padded tokens. For transferred prefixes, each context's prefix KV is
   right-padded to the longest in the batch, with a padding mask and per-row
   `position_ids` so RoPE sees each row's true positions. `oracle_contents` prefills 16
   prefixes per forward and keeps the captured KV on the GPU (`KVCapture(device=...)`).

Tests: 4 new tests check batched against one-at-a-time for store capture (stride 1 and 2)
and for standalone, oracle and store-backed scoring on mixed prefix lengths. Removing the
`position_ids` makes the oracle and store cases fail. All 61 fast and 8 slow tests pass.

Measured on the 32B:

| Phase | Old code | Batched |
|---|---:|---:|
| 14B source capture, 7,978 prefixes | ~9.5 min | 8.7 min |
| Standalone scoring, 7,978 contexts | ~85 s / 1k contexts | 687 s total (~86 s / 1k) |
| Oracle scoring, 7,978 contexts | not measured | 750 s (~94 s / 1k) |
| Zero-copy / mapper scoring, 7,978 contexts | not measured | 749 s / 848 s |

**Net effect on this run: roughly none.** Standalone was already close to compute-bound,
since each continuation re-runs the full ~130-token context. A first, larger batch budget
ran oracle at ~36 s / 1k but ran out of memory. Round-1 capture didn't benefit because
`07_eval_tasks.py` never passed `batch_size` to `build_store` (fixed below). In round 2,
where it did batch, capture was CPU-bound (~25 cores busy, 0% GPU at times): every layer's
KV is copied to the CPU in fp32 and checked, converted and saved there. **The next lever**
is doing the finiteness check and fp16 conversion on the GPU and copying only fp16 to the
CPU. The scoring budget (`eval.score_batch_tokens`) could also be tuned between 8,192
and the size that ran out of memory.

**Incidents during the switch:**
- The first batched attempt ran out of memory in oracle scoring. The token budget counted
  only each row's new tokens (~30), not its prefix cache, so one batch held ~280 rows with
  full 150-token caches across 64 layers. Fixed by budgeting transferred rows at context +
  continuation (the same size as standalone rows) and chunking the vocabulary softmax.
- The crash went through `07_eval_tasks.py`'s `finally`, which deletes the round's source
  store, so the resume re-captured the 14B round-1 store (8.7 min of GPU time lost).
- **Round 2 then ran out of memory loading the 14B, because the 32B from round 1 was still
  resident (61 GB allocated).** This is a pre-existing bug in `07_eval_tasks.py` that
  appears with any multi-round benchmark. Phase 2 kept `model` and the oracle
  generator (which references the model) as locals, so `LazyModel.release()` couldn't
  free the target. `release()` also ran `gc.collect()` while still holding its own
  reference. Both are fixed: the locals are cleared before release, and `release()`
  drops its reference before collecting. Checked on the GPU with Qwen3-0.6B: 1.23 GB →
  0.03 GB after release.
- Also fixed: `07_eval_tasks.py` now passes `runtime.batch_size` to `build_store`.

Commits: `637349b` (model leak and capture batch size), `a24dd5c` (batching and tests),
`a171632` (accelerate dependency, GPU.md H100 notes, probe verdict text).

---

## 6. Not run (out of scope for this check)

The full k-sweep, the λ sweep, the other four benchmarks (ARC-C, WinoGrande, MMLU, GSM8K)
and the 4,096-token length check. WikiText-2 perplexity was added afterwards on request
(§4.4), for the k=8 mapper only.
