# GPU Box Setup — Extra Steps

Setup commands that were needed on the validation GPU box but are not in the standard
setup instructions (README.md "Setup", NEXT_STEPS.md §2).

Box: NVIDIA A10 (23 GB), driver 570.148.08 (CUDA 12.8), Ubuntu, system Python 3.10,
1.4 TB disk, 222 GB RAM.

## 1. Install `uv`

`uv` was not preinstalled. The installer puts it in `~/.local/bin`, which is not on
`PATH` in a non-login shell:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
```

## 2. Install a torch wheel that matches the driver

`uv pip install -e ".[dev]"` resolves `torch>=2.4` to the latest PyPI wheel
(torch 2.14.0+cu130). That wheel needs a CUDA 13 driver. On a CUDA 12.8 driver it
installs without error, but `torch.cuda.is_available()` is `False`, with the warning
"The NVIDIA driver on your system is too old (found version 12080)". Reinstall torch
from the PyTorch index for the driver's CUDA version:

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# -> 2.11.0+cu128 True
```

Check the "CUDA Version" that `nvidia-smi` reports, and pick the matching `whl/cuXXX`
index.

## 3. Install `accelerate`

`src/kvtransfer/runtime.py` loads models with `device_map=...`, and transformers
requires `accelerate` for that. `accelerate` is not in `pyproject.toml`, so every
pipeline script (`02_probe_r2.py` onward) fails at model load with
"Using a `device_map` ... requires `accelerate`". The tests don't catch this because
they load on CPU without a device_map.

```bash
uv pip install accelerate      # got 1.15.0; installing it keeps torch 2.11.0+cu128
```

(Now in `pyproject.toml`'s `dependencies`, so `uv pip install -e ".[dev]"` installs it.)

## 4. A10 configs: `configs/*.a10.yaml`

The small-pair configs were sized for an 8 GB M1: `device: mps`, `batch_size: 1`,
`chunk: 5`, Grams and solves on CPU (`gram_device` unset), `fit.max_gram_gb` falling back
to 8, and no `eval.tasks`. The A10 variants keep the pair, calibration and dtypes, so KV
stores and `output_dir` are shared with the originals. Store fingerprints cover model,
token ids, stride and store dtype, not `chunk`, so an existing store is reused.

- `configs/qwen3_0.6b_to_1.7b.a10.yaml`: the full small pair
- `configs/smoke_0.6b_to_1.7b.a10.yaml`: the Step 3 smoke run

```bash
C=configs/smoke_0.6b_to_1.7b.a10.yaml
CONFIG=configs/qwen3_0.6b_to_1.7b.a10.yaml N_SEQS=500 KS="1 2 4 8 12 16" bash scripts/run_all.sh
```

| key | M1 base | A10 | why (measured on the A10 unless noted) |
|---|---|---|---|
| `runtime.device` | mps | cuda | |
| `runtime.batch_size` | 1 | 16 | 1.7B capture peak 8.5 GB (batch 8: 6.0 GB). Capture at batch 8 already takes ~1.1 min/model for 500 seqs, so this is a small win |
| `runtime.chunk` | 5 | 40 | sequences per on-disk shard (not the forward batch). Probe/fit move a whole shard to the GPU: ~2.3 GB at 40 |
| `runtime.store_dtype` | (default fp16) | float16 | explicit. Keep `dtype: float16` too: switching to bf16 would silently reuse fp16 stores |
| `probe.gram_device` | cpu | cuda | fp32 Grams. **Needs the `torch.eye` fix below or the probe crashes** |
| `fit.gram_device` | cpu | cuda | fp64 Grams (see limits below). Tied with CPU on wall time in the smoke run; frees the CPU |
| `fit.max_gram_gb` | 8 (default) | 12 | fp64 Grams: k=12 9.2 GB, k=16 16.0 GB. Measured non-Gram overhead ~4.4 GB plus ~2.3 GB shard: 12 runs k<=12 in one pass and k=16 in two. **16 would OOM**: k=16's 15.98 GB goes in one pass, ~22.7 GB against 22.6 GB free |
| `eval.store_budget_gb` | (default 50) | 100 | fewer benchmark capture rounds (0.6B source KV ~115 KB/token) |
| `eval.tasks` | unset | all five | without it `07_eval_tasks.py` runs only ARC-C + WinoGrande. Same settings as the 14B config |

The first 500-seq run used an earlier draft (`configs/a10/`, since deleted; batch 8, chunk 5,
max_gram_gb 8, no tasks). The values above have not yet run end to end at 500 seqs.

**Benchmark runtime with all five tasks.** `07_eval_tasks.py` scores every mapper in
`results/mappers/` by default: 6 k x 3 lambdas + identity x 3 + 4 baselines, ~25
configs. The smoke run scored ~60 ms per ARC context, so rough per-config estimates are
HellaSwag ~10 min, MMLU ~14 min, ARC-C + WinoGrande ~4 min, plus GSM8K generation
(5-shot, up to 256 tokens, one token at a time), likely around an hour. That's on the
order of a day for all 25 configs. Restrict it:

```bash
python scripts/07_eval_tasks.py --config $CONFIG --device cuda \
  --only standalone oracle zero_copy mean_pred mapper_identity_lam0.01 \
         mapper_k1_lam0.01 mapper_k4_lam0.01 mapper_k8_lam0.01 mapper_k16_lam0.01
```

or add `limit: N` under a task. The run is resumable (`eval_tasks.partial.json`).

Limits that config can't change:

- **The fit accumulates in fp64** (`GramAccumulator`, `src/kvtransfer/gram.py:30`, no
  config key). The A10 runs fp64 at ~1/32 of its fp32 rate, so fit time grows steeply
  with k (500 seqs: k=1 0.3 min, k=4 1.9 min, k=8 6.5 min). Whether `fit.gram_device:
  cpu` (30 cores) beats this at 500 seqs hasn't been measured.
- **Benchmark and perplexity scoring run one context at a time** (`eval/harness.py`
  batches only the continuations of one context), so the GPU is underused during
  `05_eval` / `07_eval_tasks`, and no config key batches across contexts.
- `probe.shards` in the base config is never read by the code.

Smoke-pair check: GPU Grams give R^2 identical to the CPU path to 3 decimals, and the
fit's CPU time dropped from 168 s to 15 s.

### Local bug fix needed for `probe.gram_device: cuda`

`BatchedGrid.solve_r2` / `DiagonalGrid.solve_r2` (`src/kvtransfer/batched_gram.py`)
built `torch.eye(D, dtype=f64)` with no device. With GPU Grams the probe then crashes
at the solve step: "Expected all tensors to be on the same device ... cuda:0 and cpu".
`configs/qwen3_14b_to_32b.yaml` sets `probe.gram_device: cuda`, so it hits the same
crash. Fixed and committed (`b26df04`); for reference, the fix was:

```bash
sed -i 's/eye = torch.eye(D, dtype=f64)$/eye = torch.eye(D, dtype=f64, device=cxx.device)/' \
    src/kvtransfer/batched_gram.py
```

## 5. H100 (80 GB): Qwen3-14B -> 32B

Validated 2026-09-28 on an H100 PCIe with the same driver (570.148.08, CUDA 12.8), so
steps 1-2 above apply as written. See REPORT4.md.

- `configs/qwen3_14b_to_32b_replication.yaml` runs unchanged: batch 4, chunk 25, GPU
  Grams, `max_gram_gb: 40`.
- Download both models' weights in parallel before or while the probe starts
  (`hf download Qwen/Qwen3-14B` and `Qwen/Qwen3-32B`, ~90 GB, ~100 MB/s combined here).
  The probe waits on the files and starts capturing 14B as soon as it lands.
- Timings: calibration capture ~3 min (14B) and ~3.5 min (32B); probe 12.6 min
  including the download wait; **k=8 fit ~35 s** (the H100's fp64 rate removes the A10's
  fit bottleneck); HellaSwag ~12-14 min per configuration across both rounds; WikiText-2
  perplexity ~2 min for 5 configurations.
- Peak GPU memory is ~62 GB of 32B weights plus the scoring batch. With
  `score_batch_tokens: 8192`, standalone and oracle peaked at ~72-75 GB.
- Benchmark capture is CPU-bound (~25 cores busy): fp32 KV is copied to the CPU per layer
  and checked, converted and saved there.
- `store_budget_gb: 60` gives HellaSwag 2 rounds. With ~1 TB of disk free, 100 would give
  1 round and save one 14B/32B reload.
