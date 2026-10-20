# Experiment Report 5: Two More Qwen3 Pairs — Depth vs Width (the minimum family matrix)

First extension beyond the replication of *Cross-Model KV Cache Transfer in LLM Families*
(arXiv 2608.03893). Two new matched-KV pairs, **Qwen3-4B → 8B** and **Qwen3-1.7B → 4B**,
chosen so that, together with the two pairs already measured, depth mismatch and width
mismatch can be told apart.

**Date:** 2026-10-07
**Branch:** `cross-depth-benchmarks` (on top of `fe98209`); configs and script added in this run
**Hardware:** 1× NVIDIA A100 80GB PCIe (Prime Intellect / RunPod), driver 595.91.07, 48 CPU
cores, 503 GB RAM, 120 GB local disk
**Software:** Python 3.12, PyTorch 2.11.0+cu128, transformers 5.19.0
**Configs:** `configs/qwen3_4b_to_8b.yaml`, `configs/qwen3_1.7b_to_4b.yaml`;
driver `scripts/run_pair.sh`; pod log in `results/family_matrix_pod/run_both.log`
**Protocol:** the 14B → 32B replication protocol — 500 × 1,024 FineWeb-Edu calibration,
stride 4, bf16 models, fp16 stores, all-heads mappers at k = 1 and k = 8, λ = 0.01,
held-out R² on the last 2 of 20 shards, HellaSwag only (10,042 questions, acc_norm).

Previous reports: [REPORT4.md](REPORT4.md) (14B → 32B) and the 2026-10-06 all-heads runs
recorded in the devlog.

---

## 1. Why these two pairs

All six dense Qwen3 models share 8 KV heads × 128 dims and one tokenizer, so any ordered
pair is a legal transfer. The two pairs already run differ in several things at once:
0.6B → 1.7B doubles the hidden size at equal depth; 14B → 32B changes depth (40 → 64) at
equal hidden size (5,120). The two new pairs complete a small design:

| Pair | Layers | Hidden | What changes |
|---|---|---|---|
| 0.6B → 1.7B | 28 → 28 | 1,024 → 2,048 | width only |
| **4B → 8B** | 36 → 36 | 2,560 → 4,096 | width only, mid scale |
| **1.7B → 4B** | 28 → 36 | 2,048 → 2,560 | depth and width |
| 14B → 32B | 40 → 64 | 5,120 → 5,120 | depth only |

The question: is transfer limited by depth mismatch, width mismatch, or simply scale?

---

## 2. Run

| Stage | 4B → 8B | 1.7B → 4B |
|---|---:|---:|
| Downloads | 3 min | — (4B present, 1.7B 1 min) |
| Calibration capture + probe | 10 min | 4 min |
| Fits: all-heads k=1, k=8, plus held-out refits | 5 min | 4 min |
| HellaSwag, 6 configurations | 97 min (3 rounds) | 70 min (2 rounds) |
| **Total** | **1 h 55** | **1 h 18** |

Wall time 12:43 → 15:57 UTC (3 h 14 incl. setup), about 40% over the pre-run estimate.
The benchmark's source-capture phase is CPU-bound (GPU at 0%, one Python process at
~1,900% CPU converting and writing KV), as REPORT4 found on the H100, and the 120 GB disk
forced HellaSwag into 30 GB rounds, each paying a capture and a model reload. The 8B
scoring itself ran at 10–40% GPU utilisation: at ~130 tokens per context the harness's
per-batch Python work is a large share of the time for a model this small.

Guardrails: 73 fast tests passed on the box. Oracle vs standalone: 4B → 8B +0.08 points
(p = 0.33), 1.7B → 4B −0.05 points (p = 0.59) — the graft path is exact on both targets.

Disk incident: at round 1 of the 4B → 8B benchmark the disk reached 110/120 GB (weights 24
+ two calibration stores 36 + 27 GB round store + 12 GB uv wheel cache + mappers). Freed
the 8B calibration store (not read by the eval) and the uv cache mid-run without touching
the process; the rest of the run stayed under 90 GB.

---

## 3. Results

### 3.1 Linear-structure probe (n = 500, 128K observations per head)

| | 4B → 8B | 1.7B → 4B |
|---|---:|---:|
| Best single-source key R² (content space) | 0.687 | 0.519 |
| Mean of best source per target, keys / values | 0.624 / 0.517 | 0.408 / 0.224 |
| Best source is the layer-aligned one | **36 / 36** | 23 / 36 |
| RoPE stripping (aligned mean, content vs post-RoPE) | 0.624 vs 0.634 — **no help** | 0.397 vs 0.322 — helps |

**Alignment.** 4B → 8B is a perfect diagonal: every target layer's best source is the same
index, as on 0.6B → 1.7B. 1.7B → 4B is not proportional: target layers 0–18 read source
0–17 almost one-to-one (offset 0–2), then the 4B's top 17 layers (19–35) all read source
layers 15–26, mostly 19–23. The 4B's extra depth sits at the top, and its upper third
corresponds to a narrow band of the 1.7B's upper-middle layers — the same kind of
piecewise structure REPORT4 found on 14B → 32B (target 32–60 ← source 8–36), not a stretch.

**RoPE stripping did not help on 4B → 8B** (post-RoPE keys fit slightly *better*, 0.634 vs
0.624). All Qwen3 pairs share `rope_theta`, so in the layer-aligned comparison the source
and target rotations are identical and stripping can only matter through the fit itself; it
gave +0.014 on 0.6B → 1.7B, +0.063 on 14B → 32B and +0.075 here on 1.7B → 4B, and −0.010 on
this pair. Content-space fitting remains the right choice for length generalisation, but on
a same-depth, same-θ pair it is not where the gain comes from.

### 3.2 Mapper fit (all-heads, λ = 0.01; mean over target layers and heads)

| Mapper | 4B → 8B in-sample K / V | 4B → 8B held-out | 1.7B → 4B in-sample | 1.7B → 4B held-out |
|---|---:|---:|---:|---:|
| k=1 (1,024 inputs / head) | 0.705 / 0.603 | 0.695 / 0.591 | 0.621 / 0.485 | 0.608 / 0.468 |
| k=8 (8,192 inputs / head) | 0.788 / 0.693 | **0.752 / 0.642** | 0.745 / 0.626 | **0.707 / 0.574** |

In-sample inflation at k=8 is 0.04–0.05, as on the earlier pairs. 4B → 8B has the highest
held-out R² of any pair measured so far (0.752 / 0.642, vs 0.730 / 0.615 on 14B → 32B).

### 3.3 HellaSwag (10,042 questions, acc_norm, zero-shot)

**4B → 8B**

| Configuration | acc_norm | Retention | Gap recovered |
|---|---:|---:|---:|
| Standalone (8B) | 0.7494 | 100% | — |
| Oracle graft | 0.7501 | 100.1% | — |
| **All-heads k=8** | **0.7156** | **95.5%** | **45.0%** |
| All-heads k=1 | 0.6963 | 92.9% | 13.6% |
| Zero-copy | 0.6879 | 91.8% | 0% |
| Mean prediction | 0.3227 | 43.1% | — |

**1.7B → 4B**

| Configuration | acc_norm | Retention | Gap recovered |
|---|---:|---:|---:|
| Standalone (4B) | 0.6850 | 100% | — |
| Oracle graft | 0.6845 | 99.9% | — |
| **All-heads k=8** | **0.6325** | **92.3%** | **87.6%** |
| All-heads k=1 | 0.5482 | 80.0% | 67.6% |
| Mean prediction | 0.2783 | 40.6% | — |
| Zero-copy | 0.2630 | 38.4% | 0% |

Paired tests (same questions; bootstrap CI and McNemar on discordant pairs):

| Pair | Comparison | Δ (pts) | 95% CI | p |
|---|---|---:|---:|---:|
| 4B → 8B | standalone − k=8 | +3.38 | [+2.87, +3.89] | 3e−39 |
| 4B → 8B | k=8 − zero-copy | +2.77 | [+2.15, +3.39] | 6e−18 |
| 4B → 8B | k=8 − k=1 | +1.93 | [+1.53, +2.33] | 3e−21 |
| 4B → 8B | k=1 − zero-copy | +0.84 | [+0.23, +1.44] | 0.009 |
| 4B → 8B | standalone − zero-copy | +6.14 | [+5.46, +6.84] | 8e−70 |
| 1.7B → 4B | standalone − k=8 | +5.25 | [+4.59, +5.91] | 3e−57 |
| 1.7B → 4B | k=8 − k=1 | +8.43 | [+7.75, +9.14] | 2e−124 |
| 1.7B → 4B | k=8 − zero-copy | +36.95 | [+35.74, +38.18] | ≈ 0 |

Every difference is outside noise; the smallest, k=1 over zero-copy on 4B → 8B, is +0.84 points at p = 0.009.

---

## 4. The four pairs together

| Pair | Change | Best 1-src K R² | Held-out K / V, k=8 | Zero-copy ret. | k=1 ret. | k=8 ret. | Gap recovered, k=8 | k=8 − k=1 (pts) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.6B → 1.7B | width | 0.620 | 0.710 / 0.592 | 56.1% | 89.3% | 92.2% | 82.3% | +1.8 |
| 1.7B → 4B | depth + width | 0.519 | 0.707 / 0.574 | 38.4% (chance) | 80.0% | 92.3% | 87.6% | **+8.4** |
| 4B → 8B | width | 0.687 | **0.752 / 0.642** | **91.8%** | 92.9% | 95.5% | 45.0% | +1.9 |
| 14B → 32B | depth | **0.771** | 0.730 / 0.615 | 68.4% | (paper k=1: ~54%) | **97.8%** | 93.1% | (paper: +36) |

0.6B → 1.7B and 14B → 32B rows are from the 2026-10-06 all-heads runs (devlog) and
`fe98209`. The 14B → 32B k=1 all-heads was not run here; the paper's own k=1 HellaSwag for
that pair is 44.81 acc_norm, about 54% of standalone, and it uses the all-heads features.

### 4.1 What the grid says

1. **Retention tracks scale, not depth or width mismatch.** k=8 retention rises
   monotonically with target size — 92.2% (1.7B), 92.3% (4B), 95.5% (8B), 97.8% (32B) —
   and the pure-depth pair is the *best*, not the worst. The hypothesis that depth mismatch
   is what limits the linear map is **not supported**. Held-out R² tells the same story
   less cleanly (all four land in 0.71–0.75 for keys).

2. **Depth mismatch changes the *mechanism*, not the ceiling.** On the two equal-depth
   pairs, one source layer with all heads already does nearly all the work: k=8 adds only
   +1.8 and +1.9 points over k=1. On the depth-mismatched pairs the single best layer is a
   poor substitute and multi-layer selection becomes the main ingredient: +8.4 points on
   1.7B → 4B, and the paper's +36 on 14B → 32B. Cross-layer top-k is a **depth-mismatch
   fix**; at equal depth it is close to optional.

3. **Zero-copy is where pairs differ most.** The unmapped cache retains 92% on 4B → 8B
   but 56% on 0.6B → 1.7B, both equal-depth pairs with a diagonal alignment. So the two
   width-only pairs are not alike: the 4B and 8B caches are nearly interchangeable as-is,
   and the ridge map adds only 2.8 points (45% of a small gap), while the 0.6B/1.7B caches
   need the map (+21.8 points). On unequal-depth pairs zero-copy is partly an alignment
   artefact — it pastes layer *l* into the proportional layer, which §3.1 shows is wrong
   for most layers — and on 1.7B → 4B it lands at chance.

4. **The alignment path is piecewise on every unequal-depth pair and diagonal on every
   equal-depth pair.** 28 → 36: identity for the bottom half, then the extra depth at the
   top maps to a narrow band of source layers. 40 → 64 (REPORT4): identity for 32 layers,
   then an offset of 24. Neither is the proportional stretch that `identity_alignment()`
   assumes, so that baseline is systematically wrong for unequal depths.

### 4.2 Caveats

- HellaSwag only, one calibration seed, one λ. Retention on generation tasks or long
  prefixes may order the pairs differently.
- The 4B / 8B near-compatibility is an observation, not an explanation. Whether it
  reflects shared initialisation, distillation, or something else in Qwen3's training is
  not something this data can settle.
- Zero-copy and k=1 are not comparable across equal- and unequal-depth pairs (point 3).
- 14B → 32B k=1 all-heads was not measured; the paper's value stands in for it.

---

## 5. Not run

The other four benchmarks, any k between 1 and 8, λ ≠ 0.01, downward transfer, chained
transfer (0.6B → 1.7B → 4B vs direct), and the 4,096-token length check. The fitted k=8
all-heads mappers for both pairs were pulled off the pod
(`results/<pair>/mappers/mapper_k8_lam0.01_allheads.pt`, 2.3 GB each) for a later
mapper-anatomy analysis; the HellaSwag source stores were not kept (120 GB disk).

## 6. Reproducing

```bash
# on an 80 GB A100 with >=120 GB local disk; GPU.md §1–3 for the environment
bash scripts/run_pair.sh configs/qwen3_4b_to_8b.yaml
rm -rf results/kv/Qwen__Qwen3-8B            # free 18 GB before the second pair
bash scripts/run_pair.sh configs/qwen3_1.7b_to_4b.yaml
```

Both configs cap each benchmark round at 30 GB of source KV for the small disk; with
200 GB or more, raise `eval.store_budget_gb` to 80 and set `keep_stores: true`.
