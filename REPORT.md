# Experiment Report

Replication of *Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear
Mapping for Prefill Reuse* (arXiv 2608.03893, Heo et al., NVIDIA) on **Qwen3-0.6B →
Qwen3-1.7B**.

**Date:** 2026-09-18
**Hardware:** Apple M1, 8 GB unified memory, macOS 14.2.1
**Software:** Python 3.12.13, PyTorch 2.14.0, transformers 5.17.0

---

## 1. Summary

The implementation is complete and verified. **The headline R² and retention numbers
were not produced**, because the development machine cannot hold Qwen3-1.7B alongside a
normal desktop session. This report records what was verified, what was measured, and
what the remaining run requires.

Three findings are already solid and independent of the blocked run:

1. **The graft path is exact.** Capturing content-space KV and re-applying RoPE
   reproduces the model's own cache with **0.0 error** across all 28 layers; grafting a
   prefix cache and decoding from it reproduces ordinary teacher-forced logits to
   **2.9e-05**. Any later transfer number is therefore measuring the *mapping*, not
   harness error.
2. **Qwen3.5 0.8B/2B cannot be used for this method** (Section 3).
3. **The paper's closed form is genuinely laptop-scale** once the statistics are
   accumulated correctly — the working set is ~0.9 GB, not the gigabytes a naive
   implementation needs (Section 5).

---

## 2. Model pair selection

The method requires a **matched-KV** pair: identical KV head count and per-head
dimension, dense full attention, and a shared tokenizer (the regression pairs KV vectors
token by token, so both models must see identical token streams).

Verified from the published `config.json` of each model:

| | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|---|---|---|---|
| `num_hidden_layers` | 28 | 28 | 36 |
| `num_key_value_heads` | 8 | 8 | 8 |
| `head_dim` | 128 | 128 | 128 |
| `hidden_size` | 1024 | 2048 | 2560 |
| `rope_theta` | 1e6 | 1e6 | 1e6 |

0.6B → 1.7B is matched-KV with a 2.8× parameter ratio, inside the paper's 2.3–8.8×
range. **Equal depth is a property no pair in the paper has**, which removes the
layer-alignment confound and supplies a clean identity-alignment baseline (layer *l* →
layer *l*) against which cross-layer top-k selection can be judged.

1.7B → 4B is available as a second matched-KV pair that crosses a depth boundary
(28 → 36), if the first pair yields a positive result.

---

## 3. Negative result: Qwen3.5 0.8B → 2B is out of scope

Recorded so it is not rediscovered. Both `Qwen/Qwen3.5-0.8B` and `Qwen/Qwen3.5-2B`
declare:

```json
"layer_types": ["linear_attention", "linear_attention", "linear_attention",
                "full_attention", ...],
"full_attention_interval": 4,
"num_key_value_heads": 2,
"head_dim": 256
```

Only **6 of 24 layers are full attention**; the remaining 18 carry recurrent state
rather than a KV cache, so there is nothing to transfer for three quarters of the model.
They are also 2 × 256 rather than 8 × 128, so they are not matched-KV against Qwen3 in
either direction. The paper scopes explicitly to dense full-attention models, and
end-to-end retention is not a meaningful measurement here.

---

## 4. Verified correctness

`pytest tests/` — **20 fast tests + 4 end-to-end guardrails, all passing.**

### 4.1 RoPE convention (`tests/test_rope.py`, 6 tests)

Qwen3 uses the **non-interleaved (GPT-NeoX `rotate_half`)** convention, which pairs
dimension *i* with *i + D/2*, not the interleaved GPT-J form that pairs adjacent
dimensions. Using the wrong one still runs and still produces plausible-looking numbers
while silently corrupting every key, so the implementation is checked against HF's own
`apply_rotary_pos_emb` rather than against itself. A dedicated test asserts the two
conventions genuinely differ, so a future substitution would be caught.

Content-space round-trips were verified exact (1e-16) at positions 0, 1024, 8192 and
32768 — the property the method depends on for generalizing beyond the calibration
length.

### 4.2 Hook placement

Qwen3's attention applies RMSNorm to keys **before** RoPE:

```python
key_states = self.k_norm(self.k_proj(h).view(shape)).transpose(1, 2)
q, k = apply_rotary_pos_emb(q, k, cos, sin)      # rotation happens here
past_key_values.update(k, v, layer_idx)          # cache stores POST-RoPE keys
```

Content space is therefore the output of `k_norm` — not `k_proj`, and not the cache.
Measured: re-applying our RoPE to captured keys reproduces the model's own cache with
**max error 0.000e+00** on keys and values across all 28 layers.

### 4.3 End-to-end guardrail (`tests/test_identity.py`, 4 tests)

Grafting a model's own cache back into itself must be a no-op. Measured: grafted logits
match standalone teacher-forced logits to **2.9e-05**. This catches wrong RoPE
convention, wrong hook site, GQA head misalignment, position-id off-by-one, and cache
layout transposition in one test.

Two incidents during development illustrate why it exists:

- The first version of the test failed with a 25.8 error. The values were correct but
  shifted by one index — a bug in the *test's* reference slice, not the graft. Confirmed
  empirically by scanning offsets and finding an exact match (2.9e-05) at `full[:, P:]`.
- An identity fit produced R² = 1.0 with a non-identity `W`. Diagnosis: the test text
  was one paragraph repeated six times, giving **rank 127 of 128** with only 42 unique
  tokens. Under rank deficiency infinitely many `W` achieve R² = 1 and ridge returns the
  minimum-norm one. Varied text gives full rank 128. The assertion was changed to test
  that the map *acts* as the identity on real data, since an exact-`W` assertion tests
  the conditioning of the sample rather than the correctness of the fit.

### 4.4 Pipeline stages on real activations

Beyond the unit tests, the Stage A path was exercised on real Qwen3-0.6B activations
using a 4-layer subset (layers 0, 7, 14, 21) so the statistics stay small enough to run
on the development machine. Using the model as both source and target, the fit must
recover the identity:

| Stage | Result |
|---|---|
| capture → `BatchedGrid` → `solve_r2` | self-R² per layer = **1.0000, 0.9921, 0.9946, 0.9942** |
| `top_k(r2, 1)` source selection | `{0:[0], 1:[1], 2:[2], 3:[3]}` — each layer picks itself |

Both are correct: a layer predicts itself essentially perfectly, and selection recovers
the identity alignment. The slight shortfall below 1.0 on layers 7/14/21 is float32
accumulation with λ=1e-6 on finite, correlated real activations, not an error — the
synthetic-data tests confirm exact recovery under the same code path.

The remaining stages (mapper fit → graft → perplexity) are covered by the unit tests and
the end-to-end guardrail in §4.3, but were not run as a single script on this machine;
the 1.7B-sized run is what §5 explains is blocked.

### 4.5 Batched statistics (`tests/test_batched_gram.py`, 6 tests)

The batched `einsum` grid is asserted equal to the reference per-cell accumulator to
**1e-6** on every cell, plus chunked-accumulation equivalence, the diagonal special
case, identity recovery, and float32-vs-float64 agreement.

---

## 5. Performance and memory findings

Measured on the M1 (1024-token sequence, single batch):

| | load | forward |
|---|---|---|
| Qwen3-0.6B, MPS fp16 | 6 s | 2.7 s |
| Qwen3-1.7B, MPS fp16 | 75 s | 34–45 s |
| Qwen3-1.7B, CPU fp32 | 71 s | 105–127 s |

MPS is ~2.3–3.7× faster than CPU and works correctly with forward hooks (an early
hypothesis that hooks deadlock on Metal was tested and **disproved** — all four hook
patterns completed normally).

### Three real bugs, found and fixed

| Bug | Symptom | Fix |
|---|---|---|
| `shuffle(buffer_size=2000)` on a streaming dataset | 12 min and 270 MB to fetch **4 sequences** — the buffer must fill before yielding anything | take a contiguous run and vary `seed` by skipping; **6.8 s** |
| Statistics re-captured once per shard | 4× redundant model forwards; a full run projected at **19 h** instead of 5 h | one capture feeds every shard |
| 12,768 per-cell accumulators; `permute().reshape()` copies held six at a time | **6.9 GB** peak and severe heap fragmentation | batched `einsum` grid in 5 contiguous tensors; **2.4 GB**, and statistics for the full 28×28×8 grid fit in **0.88 GB** |

The third deserves emphasis: `permute(...).reshape(...)` cannot return a view, so each
call materializes a full copy. Holding all six captured tensors plus their six reshaped
views doubled peak memory for no benefit.

A separate observation worth recording: `xtx` does not depend on the target layer — every
target regresses on the same source features — so it is stored once per (source layer,
head) rather than per cell, a 28× saving on the largest array.

### Why the run is blocked

The machine runs at **~0.06 GB free with 9–12 GB of swap in use** before this project
starts: VS Code 1.69 GB, Chrome 0.89 GB, editor 0.63 GB. Qwen3-1.7B needs ~3.4 GB
contiguous. Loaded alone it is fine (33 s, 22.7 s forward), but there is no room
alongside a normal session, and the process enters uninterruptible sleep (`UN` state)
thrashing on swap rather than computing. Several runs were killed in this state; the
symptom looks like a hang but is paging.

Disk was also reclaimed during this work: **~4 GB** from the Hugging Face Xet chunk
store (which silently keeps a second copy of every blob — it held 2.6 GB while the
visible cache directory reported 4.3 MB), ~6 GB of unused cached models, and ~1.5 GB of
uv/pip/dataset caches.

---

## 6. What has not been measured

The R² probe, the k-sweep, and the retention numbers. These need one run on a machine
with ~6 GB free; see README §"Running it elsewhere". Expected duration on an A10/L4 is
30–40 minutes for the paper-faithful 500-sequence configuration.

The comparison the run should produce:

| Quantity | Paper (Qwen3 14B→32B) | This pair (0.6B→1.7B) |
|---|---|---|
| Best single-source K R² | 0.81 | — |
| Multi-source ridge R² (K / V) | 0.79 / 0.65 | — |
| Accuracy retention | 97.6 % at k=8 | — |

### A caveat to state in any writeup

Both models share `rope_theta = 1e6` and `head_dim = 128`, so `R_s == R_t` and the
strip/re-apply cancels at the calibration positions. Stripping still earns its place —
it makes the fitted `W` position-independent, which is what allows a mapper fitted at
positions 0–1023 to work at position 4096, and the length-generalization check at 4096
tests exactly that. But **this pair does not validate the cross-RoPE-config part of the
paper's contribution**, and results should not be reported as if it did.

### Baselines that make the retention number honest

The eval harness computes four, because a transfer perplexity alone is uninterpretable:

- **target standalone** — the ceiling
- **oracle graft** — target's own cache through the graft path; must match the ceiling,
  otherwise the harness is at fault rather than the mapper
- **zero-copy** — source cache pasted in unmapped; answers "is the mapper doing anything
  at all?"
- **mean prediction** — per-channel means, the R² = 0 floor; a cache of means is not
  noise and can still score above chance on multiple-choice tasks

**A negative result here is a real finding.** The paper's smallest pair is 3B and nothing
below that has been reported. If 0.6B and 1.7B do not share enough linear structure, that
is publishable information about the scale-dependence of the claim, and this repository
is built to measure it rather than to confirm it.
