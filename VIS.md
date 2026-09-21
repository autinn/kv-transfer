# Visualization Guide

The plots in `results/plots/` summarize the Qwen3-0.6B -> Qwen3-1.7B KV-cache
transfer experiment. They answer different questions, so they should not all be
read as direct measures of generation quality.

## Read the plots in this order

1. `perplexity_comparison.png` — does the transfer work end to end?
2. `perplexity_vs_k_lambda.png` — which mapper configuration works best?
3. `probe_r2_heatmaps.png` — where do the source and target representations line up?
4. The remaining plots — how stable and interpretable is that result?

## End-to-end quality

### `perplexity_comparison.png`

This is the most important plot. The y-axis is perplexity on held-out WikiText-2
continuations, shown on a logarithmic scale. Lower is better.

- `target_standalone`: ceiling. The target model builds its own prefix cache.
- `oracle_graft`: control for the cache grafting machinery. It should nearly equal
  `target_standalone`; otherwise the evaluation harness is broken.
- `zero_copy`: source cache pasted into the target without mapping.
- `mean_pred`: a deliberately weak R^2 = 0 floor.
- `mapper_*`: learned source-to-target cache mappings.

For the current run, standalone is about 16.7, the best mapper is about 29.0, and
zero-copy is about 59.9. The mapper therefore helps substantially, but does not
match the target's own cache.

### `perplexity_vs_k_lambda.png`

This shows the mapper sweep. The x-axis is `k`, the number of source layers used for
each target layer. Each line is one ridge penalty `lambda`. Lower perplexity is
better. A curve that gets worse as `k` grows indicates that extra features are not
helping end-to-end, even if the training fit improves.

The current best tested point is `k=8, lambda=0.01`.

## Representation structure

### `probe_r2_heatmaps.png`

Each panel is a source-layer by target-layer matrix. Rows are source layers and
columns are target layers. Color is R^2 from a single-source ridge fit; brighter
means the source activation predicts the target activation better.

- Content-space keys are the main probe because RoPE has been removed.
- Values have no RoPE and provide an independent comparison.
- Post-RoPE keys show what happens without stripping position-dependent rotation.

A bright diagonal means corresponding layers have similar linear structure. Bright
off-diagonal regions would suggest that cross-layer selection is useful. R^2 here is
fit quality on the calibration data, not perplexity and not proof of useful cache
transfer by itself.

### `best_source_layer_alignment.png`

The dashed line is perfect layer alignment: target layer `l` uses source layer `l`.
The solid line is the highest-R^2 source layer for each target layer.

When the lines overlap, the models' corresponding layers are the best single-source
matches. In this run all 28 target layers are layer-aligned, so top-k cross-layer
selection has little evidence to exploit at the single-source level.

### `probe_r2_distributions.png`

This compares R^2 across all source-target layer pairs with the aligned pairs only.
The box shows the middle half of the data, the center line is the median, and the
marker is the mean. It reveals whether a headline average is representative or is
driven by a small number of unusually good pairs.

### `probe_r2_by_head.png`

Each box is one KV head. It shows whether the cross-model relationship is consistent
across heads. Large differences between boxes suggest that a mean over heads hides
head-specific behavior.

## Mapper diagnostics

### `mapper_fit_r2.png`

This reports the mean in-sample R^2 stored inside each fitted mapper. Keys and values
are shown separately. Higher is better as a regression fit, but this plot is not a
substitute for perplexity: a mapper can fit calibration activations better and still
generalize worse.

### `mapper_r2_vs_k.png`

This isolates the regression fit as the number of source layers increases. It helps
diagnose capacity and regularization. Compare it with `perplexity_vs_k_lambda.png`:
if R^2 rises while perplexity worsens, the mapper is fitting structure that does not
survive end-to-end use.

### `mapper_source_selections.png`

Each row corresponds to a target layer and each column is one selected source-layer
slot. The color identifies the source layer index. Read horizontally to see which
source layers feed a target; compare panels to see what changes as `k` grows.

This is an explanatory plot, not a quality metric.

## Important limitations

- Probe R^2 and mapper fit R^2 are in-sample regression statistics.
- The current probe used 100 calibration sequences, not the paper's 500-sequence run.
- End-to-end perplexity is the strongest result because it uses held-out WikiText-2.
- `oracle_graft` should match `target_standalone`; a large gap invalidates mapper
  conclusions until the graft path is fixed.
- A lower perplexity is better, while a higher R^2 is better. Do not compare their
  numeric values directly.