# Experiment Report: Latest Results

Replication of *Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear
Mapping for Prefill Reuse* on **Qwen3-0.6B -> Qwen3-1.7B**.

**Updated:** 2026-09-21  
**Pair:** Qwen/Qwen3-0.6B -> Qwen/Qwen3-1.7B  
**Architecture:** 28 layers, 8 KV heads, head dimension 128, shared tokenizer and
`rope_theta = 1e6`

## 1. Executive summary

The cross-model experiment has now been completed at the 100-sequence scale.
The two models contain meaningful but imperfectly aligned KV representations:

- Best single-source key fit: **R^2 = 0.625**
- Layer-aligned mean key fit: **R^2 = 0.537**
- Layer-aligned mean value fit: **R^2 = 0.408**
- Best mapper evaluation: **29.04 perplexity** at `k=8`, `lambda=0.01`
- Target standalone evaluation: **16.70 perplexity**
- Zero-copy baseline: **59.85 perplexity**

The result is positive but partial. A learned mapper is substantially better than
copying the source cache unchanged, but it does not reproduce the target model's own
cache. The method therefore transfers useful information at this scale, while losing
enough information to leave a large quality gap.

## Initial project proposal

### Project and topic

This project is an experiment and partial reproduction in efficient language-model
inference. The question is: **can a smaller language model's prefill KV cache be
mapped into the format expected by a larger model, allowing the larger model to skip
most of its prompt processing?** The project reproduces the central idea of *Cross-
Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill
Reuse* at a smaller scale, using Qwen3-0.6B as the source model and Qwen3-1.7B as the
target model.

The two deep-learning topics are transformer inference and representation transfer.
The project examines what a KV cache actually represents, why keys require careful
handling of rotary position embeddings (RoPE), and whether internal representations
from two models in the same family are related by simple linear maps. It also studies
the difference between fitting an activation-level regression and improving the final
language-model predictions. This is more useful than treating inference as a black
box: the implementation exposes attention activations, accumulates regression
statistics, fits the maps in closed form, and grafts the resulting cache back into the
target model.

The initial ideas were broadly about inference, an Unsloth notebook, and a possible
LinkedIn demonstration. They were narrowed into a measurable research experiment.
The resulting artifact is more substantial than a notebook demo: it has a small-model
smoke test, correctness guardrails, a reproducible calibration pipeline, baselines,
plots, and an end-to-end evaluation. A notebook or short demo can still be added later
as a presentation layer, but the experiment is the primary contribution.

### Research background and sources

The main source is Heo et al., *Cross-Model KV Cache Transfer in LLM Families: A
Closed-Form Linear Mapping for Prefill Reuse* (arXiv:2608.03893). That paper motivates
per-head ridge regression from source KV vectors to target KV vectors and reports
substantial accuracy retention and speedups for larger model pairs. This project tests
whether the claimed linear structure survives for a 0.6B -> 1.7B pair, below the
paper's reported model sizes.

The implementation also relies on the Qwen3 model and configuration documentation to
verify matched KV-head counts, head dimensions, tokenizer compatibility, and RoPE
settings. The relevant technical ideas are standard transformer attention, KV-cache
decoding, ridge regression, and the RoPE formulation used by the Hugging Face Qwen3
implementation. These sources are important because a cache is position-dependent:
fitting directly on post-RoPE keys could learn the calibration positions rather than a
position-independent content transformation. The project therefore captures keys
before RoPE, fits in content space, and reapplies the target model's RoPE only when
building the target cache.

### Evaluation plan

The primary research measurement is held-out continuation perplexity. Calibration
sequences come from FineWeb-Edu, while evaluation sequences come from WikiText-2 so
that the mapper is tested outside its fitting domain. The target model's standalone
perplexity is the quality ceiling. An oracle graft, in which the target's own cache is
sent through the grafting path, checks that the evaluation machinery itself is sound.
The main baselines are zero-copy transfer, which pastes the source cache into the
target unchanged, and mean prediction, which ignores the source and predicts target
activation means. A useful mapper must beat zero-copy and should approach the oracle.

The secondary measurement is activation-level R^2 for keys and values. A probe fits
single-source maps for every source layer, target layer, and KV head. Production maps
then concatenate the top-k source layers and sweep the ridge penalty. Success was
defined before the final run as: (1) the identity and oracle guardrails must pass; (2)
the learned mapper must beat zero-copy on held-out perplexity; (3) increasing model
capacity should be interpreted using held-out results rather than in-sample R^2; and
(4) negative or weak transfer should be reported rather than hidden.

### Iteration plan and constraints

The project was designed to run on a laptop or a free GPU tier within hours rather
than weeks. The iteration plan was:

1. Implement and test RoPE transformations, activation capture, streaming Gram
  statistics, and the closed-form ridge solve on synthetic data.
2. Run a self-to-self Qwen3-0.6B smoke test. The fitted map should act as the identity,
  and an oracle cache should reproduce standalone logits.
3. Run the cross-model R^2 probe on a small calibration set before spending compute on
  production fitting. This decides whether the research question is viable.
4. Fit a small `k` and regularization sweep, evaluate against all baselines, and plot
  both R^2 and perplexity.
5. Scale the promising configuration to more calibration sequences, test longer
  prefixes, and document negative results and failure modes.

The completed work follows this sequence. It demonstrates that a small model pair does
share transferable KV structure, but that better regression fit does not guarantee
better language-model behavior. That gap is itself part of what the project set out to
understand: efficient inference depends not only on matching activations numerically,
but on preserving the information the target model needs for next-token prediction.

## 2. Experimental setup

### Probe and fitting

- Calibration data: FineWeb-Edu
- Calibration size: 100 sequences
- Sequence length: 1,024 tokens
- Token subsampling stride: 4
- Probe regularization: `lambda = 0.01`
- Probe output: `results/probe_n100.pt`
- Probe report: `results/probe_n100_report.txt`

The probe fits a separate ridge map for every source layer, target layer, and KV head.
Keys are fitted in content space with RoPE removed; values do not require RoPE
handling. The production mappers concatenate the top `k` source layers selected by
the probe and fit separate key and value maps per target layer and head.

### End-to-end evaluation

- Evaluation data: WikiText-2
- Evaluation size: 64 sequences
- Prefix length: 512 tokens
- Metric: perplexity on the continuation after the prefix
- Evaluation output: `results/eval_ppl.json`

This is a held-out domain relative to the FineWeb-Edu calibration set. Lower
perplexity is better.

## 3. Linear-structure probe

The probe found a clear layer-aligned pattern. The best source layer for every one of
the 28 target layers was the corresponding source layer: **28/28 aligned selections**.
This means the models' layer correspondence is strong enough that top-k selection does
not need to search far from the diagonal for this equal-depth pair.

| Quantity | Best R^2 | Layer-aligned mean |
|---|---:|---:|
| Content-space keys | 0.625 | 0.537 |
| Post-RoPE keys | 0.618 | 0.523 |
| Values | 0.509 | 0.408 |

Stripping RoPE improves the aligned key mean from **0.523** to **0.537**. The gain is
modest here because both models use the same RoPE configuration, but content-space
fitting remains the position-independent formulation required by the method.

The key result is below the paper's reported best single-source reference of 0.81 for
Qwen3-14B -> 32B, but it is clearly non-trivial. The smaller pair shares internal
structure without matching as closely as the larger pair reported in the paper.

The heatmaps and distributions are in:

- `results/plots/probe_r2_heatmaps.png`
- `results/plots/best_source_layer_alignment.png`
- `results/plots/probe_r2_distributions.png`
- `results/plots/probe_r2_by_head.png`

## 4. Mapper fitting

The saved mappers sweep `k` in `{1, 2, 4, 8, 12, 16}` and `lambda` in
`{0.001, 0.01, 0.1}`. Mean in-sample fit quality rises with `k`, but that does not
guarantee better end-to-end behavior.

| Mapper | Key R^2 | Value R^2 |
|---|---:|---:|
| `k=1`, `lambda=0.001` | 0.541 | 0.408 |
| `k=2`, `lambda=0.001` | 0.586 | 0.455 |
| `k=4`, `lambda=0.001` | 0.630 | 0.505 |
| `k=8`, `lambda=0.001` | 0.672 | 0.557 |
| `k=12`, `lambda=0.001` | 0.694 | 0.584 |
| `k=16`, `lambda=0.001` | 0.711 | 0.606 |

The corresponding visualizations are in:

- `results/plots/mapper_fit_r2.png`
- `results/plots/mapper_r2_vs_k.png`
- `results/plots/mapper_source_selections.png`

## 5. End-to-end perplexity

The target model evaluated its own prefix cache, an oracle cache through the graft
path, the learned mappers, and two controls.

| Configuration | Perplexity |
|---|---:|
| Target standalone | **16.699** |
| Oracle graft | **16.700** |
| Mapper `k=8`, `lambda=0.01` | **29.043** |
| Mapper `k=4`, `lambda=0.001` | 29.476 |
| Mapper `k=2`, `lambda=0.001` | 31.436 |
| Mapper `k=1`, `lambda=0.001` | 31.602 |
| Identity mapper, `lambda=0.001` | 31.602 |
| Zero-copy | 59.855 |
| Mean prediction | 628.090 |

The full sweep is recorded in `results/eval_ppl.json` and visualized in:

- `results/plots/perplexity_comparison.png`
- `results/plots/perplexity_vs_k_lambda.png`

The oracle graft matches the standalone target to 0.001 perplexity, so the graft and
evaluation path is not responsible for the mapper's quality gap. The best mapper is
about twice as good as zero-copy in perplexity terms, although perplexity is
exponential and should not be interpreted as a simple information-retention ratio.

The important generalization result is that larger `k` improves calibration R^2 but
does not keep improving held-out perplexity. `k=16` has a better in-sample fit than
`k=8`, yet its evaluated perplexity is much worse. This indicates that additional
source features increase capacity faster than they improve transferable structure,
or that the current calibration size is insufficient for the larger maps.

## 6. Interpretation

This experiment supports three conclusions:

1. **Cross-model KV structure exists below the paper's reported 3B scale.** The
   non-trivial probe R^2 and the mapper's improvement over zero-copy support this.
2. **Layer alignment is the dominant structure for this equal-depth pair.** All target
   layers selected their corresponding source layer as the best single source.
3. **Representation similarity does not translate perfectly to cache usability.** The
   best mapper remains well above target standalone perplexity, and in-sample R^2 is
   not a reliable optimizer for end-to-end quality.

The result is not a replication of the paper's reported retention number. The paper's
accuracy-retention benchmark was not run here; this report uses held-out perplexity as
the available sensitive transfer metric.

## 7. Limitations and next experiments

- The probe uses 100 calibration sequences rather than the paper-faithful 500.
- The mapper fit statistics are in-sample; the end-to-end perplexity evaluation is
  the stronger generalization check.
- Only the `k` values that have saved mapper files were evaluated: 1, 2, 4, 8, 12,
  and 16. The configuration also mentions larger values that were not present in the
  current output directory.
- This pair shares `rope_theta` and head dimension, so it does not test transfer
  across different RoPE configurations.
- A 4,096-token prefix evaluation would test the position-independent claim directly;
  no such result is included in the current artifact set.
- Accuracy retention on a downstream benchmark remains unmeasured.

The highest-value follow-up is a 500-sequence probe and mapper fit, followed by
repeating the held-out evaluation and the 4,096-token length-generalization check.