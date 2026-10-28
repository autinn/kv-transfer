# Cross-Model KV Cache Transfer in the Qwen3 Family

A replication and extension of Heo et al. (NVIDIA), **"Cross-Model KV Cache Transfer in LLM
Families: A Closed-Form Linear Mapping for Prefill Reuse"**
([arXiv 2608.03893](https://arxiv.org/html/2608.03893v1)).

**The idea:** a small model reads the prompt. A linear map turns its KV cache into the format a
larger model from the same family expects. The larger model then answers without reading the
prompt itself. The map is a ridge regression per layer and KV head, solved in closed form.

**What this repo adds:** the method tested on four Qwen3 pairs, from 0.6B → 1.7B (smaller than
anything in the paper) to the paper's own 14B → 32B.

---

## Results

HellaSwag, all 10,042 questions, zero-shot, acc_norm. **Retention** is the large model's
accuracy with the mapped cache, divided by its accuracy on its own. All mappers use the same
protocol: 500 × 1,024 FineWeb-Edu calibration tokens, all-heads features, λ = 0.01.

| Pair | Layers | Hidden size | Zero-copy | k = 1 | **k = 8** | Held-out key R² (k = 8) |
|---|---|---|---:|---:|---:|---:|
| 0.6B → 1.7B | 28 → 28 | 1,024 → 2,048 | 56.1% | 89.3% | **92.2%** | 0.710 |
| 1.7B → 4B | 28 → 36 | 2,048 → 2,560 | 38.4% | 80.0% | **92.3%** | 0.707 |
| 4B → 8B | 36 → 36 | 2,560 → 4,096 | 91.8% | 92.9% | **95.5%** | 0.752 |
| 14B → 32B | 40 → 64 | 5,120 → 5,120 | 68.4% | not run | **97.8%** | 0.730 |

- **k** is how many small-model layers feed each large-model layer.
- **Zero-copy** hands the large model the small model's cache unmapped. The mapper has to beat it.
- Every k = 8 vs zero-copy and k = 8 vs k = 1 gap is significant on a paired test
  (`scripts/08_paired_test.py`). The smallest is +1.8 points (p = 3e−14).

Findings:

1. **The method replicates.** On 14B → 32B, k = 8 scores 80.93 acc_norm; the paper reports 80.70.
2. **It works below 3B.** Both small pairs keep about 92%.
3. **At equal depth, one layer does almost all the work.** k = 8 adds about 2 points over k = 1.
   When depth changes (1.7B → 4B), it adds 8.4.
4. **14B → 32B has a head start.** Its `k_proj` weights show the 32B's layers are copies of the
   14B's: layers 0–31 match 14B layers 0–31, and 32–63 match 8–39 (Fig 7). Part of its 97.8%
   may come from that shared origin rather than from model size.

![HellaSwag retention for the four pairs](results/plots/writeup/fig1_retention.png)

### Speedup

Time for the large model to get a usable cache: map the small model's cache vs re-read the
prompt. Paper protocol: 64 to 32K tokens, 50 warmup and 30 timed runs, median
(`scripts/10_speedup.py`).

| Pair | GPU | 64 tokens | 1K | 8K | 32K | Paper |
|---|---|---:|---:|---:|---:|---|
| 14B → 32B, k = 8 | H100 PCIe | 5.6× | 13.1× | 15.7× | **21.8×** | 4.4× / 17× / 25.1× |
| 0.6B → 1.7B, k = 8 | L4 | 2.6× | 2.3× | 2.7× | out of memory | not reported |

These numbers assume the small model has already read the prompt. Counting its prefill too,
14B → 32B is 1.6–2.1× faster end to end. Timing uses a mapper of the fitted shape with random
weights, which does the same arithmetic as a fitted one.

---

## Run it yourself

### Setup

```bash
git clone https://github.com/autinn/kv-transfer && cd kv-transfer
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
source .env.sh                      # HF_HUB_DISABLE_XET=1: avoids a second copy of every download
pytest tests/ -q -m "not slow"      # 78 passed, no downloads
```

### 1. Laptop trial: test the code on a Mac or CPU

The whole method at small scale on 0.6B → 1.7B. It calibrates on 64 sequences of 512 tokens
(the full run uses 500 of 1,024): it captures both models' KV, runs the probe, fits one mapper
(k = 1) and scores it on 200 HellaSwag questions.

```bash
bash scripts/run_laptop.sh              # Apple silicon
DEVICE=cpu bash scripts/run_laptop.sh   # any machine
```

Needs about 5 GB of downloads and 8 GB of RAM. On an 8-core cloud CPU (`DEVICE=cpu`) it took
20 minutes: about 7 to download, capture, probe and fit, and 13 to score the four configurations.
A laptop will be faster or slower depending on its cores.

Expected result on CPU (MPS runs in float16, so its numbers can differ by a question or two):

| Config | HellaSwag acc_norm (200 questions) | Retention |
|---|---:|---:|
| standalone | 0.495 | 100% |
| oracle | 0.495 | 100% |
| zero_copy | 0.295 | 59.6% |
| mapper_k1_lam0.01_allheads | 0.440 | 88.9% |

What to check:

- **Oracle = standalone.** The oracle passes the large model's own cache through the transfer
  path. Any gap means the transfer code is broken.
- **Mapper > zero-copy.** The fitted map carries real information.

Retention lands close to the full run's k = 1 figure (89.3%), but 200 questions make it
noisy: each accuracy is ±3–4 points, so treat it as a working check, not a result.

### 2. Full small run: 0.6B → 1.7B on one GPU (about 1.5 h)

This produces the 0.6B → 1.7B row above. It was run on Modal:

```bash
pip install modal && modal setup
modal deploy modal_run.py
modal run modal_run.py::spawn_check --k 1                    # L4, under 90 min, about $1
modal run modal_run.py::spawn_check --k 8 --gpu A100-40GB    # reuses the probe; adds k = 8
```

The first command captures both models' KV, runs the probe, fits k = 1 and scores HellaSwag
for standalone, oracle, zero-copy and k = 1. The second fits and scores k = 8. Fetch the
results with `modal volume get kv-transfer-cache results/eval_tasks.json .`

On your own GPU (24 GB or more, about 100 GB free disk), the same stages run with
`bash scripts/run_pair.sh configs/qwen3_0.6b_to_1.7b.a10.yaml`.

Expected (committed as `results/qwen3_0.6b_to_1.7b/eval_tasks.json`):

| Config | HellaSwag acc_norm | Retention |
|---|---:|---:|
| standalone | 0.6031 | 100% |
| oracle | 0.6031 | 100% |
| zero_copy | 0.3381 | 56.1% |
| mapper_k1_lam0.01_allheads | 0.5384 | 89.3% |
| mapper_k8_lam0.01_allheads | 0.5563 | 92.2% |

### 3. Larger pairs (one 80 GB GPU, 120 GB or more of disk)

```bash
bash scripts/run_pair.sh configs/qwen3_1.7b_to_4b.yaml   # about 1.3 h
bash scripts/run_pair.sh configs/qwen3_4b_to_8b.yaml     # about 2 h
```

Each run goes probe → fit (k = 1 and 8) → held-out R² → HellaSwag → paired test, and resumes
from whatever is already on disk. For 14B → 32B, use
`configs/qwen3_14b_to_32b_replication.yaml` (see [GPU.md](GPU.md)).

### Fitted mappers

The mappers are 0.24–2.4 GB each, too large for git. They're on Hugging Face at
[autinnn/qwen3-kv-transfer-mappers](https://huggingface.co/autinnn/qwen3-kv-transfer-mappers).
Score one without refitting:

```bash
python scripts/07_eval_tasks.py --config configs/qwen3_4b_to_8b.yaml --tasks hellaswag \
  --mappers <downloaded .pt>
```

### Figures

`python scripts/09_writeup_figures.py` rebuilds every figure in `results/plots/writeup/` from
the committed results, with no GPU, in about a minute.

| Figure | What it shows |
|---|---|
| `fig1_retention.png` | Retention per pair: zero-copy, k = 1, k = 8 |
| `fig2_alignment.png` | Best small-model layer for each large-model layer |
| `fig3_probe_heatmaps.png` | Single-layer key R² for every layer pair |
| `fig4_heads.png` | Same-head vs all-heads features |
| `fig5_heldout_r2.png` | Fit on training text vs held-out text |
| `fig6_speedup.png` | Latency and speedup vs prompt length |
| `fig7_lineage.png` | 14B → 32B: probe R² next to the cosine of the raw `k_proj` weights |

---

## Method

For each large-model layer and KV head, with RoPE removed from the keys:

```
K̂ = (K_s · R_s⁻¹ · W_K + b_K) · R_t      keys: un-rotate, map, re-rotate
V̂ =  V_s · W_V + b_V                     values carry no position
W = (XᵀX + λI)⁻¹ XᵀY,  λ = 0.01           X = all 8 KV heads of the top-k small-model layers
```

1. **Probe.** Fit a one-layer map for every (small layer, large layer, head) and keep only its
   R². This ranks which small-model layers to read from.
2. **Fit.** For each large-model layer, take the top-k layers from the probe and solve the ridge
   regression from `XᵀX` and `XᵀY`, accumulated shard by shard.

Two details that are easy to get wrong, both covered by tests:

- **RoPE convention.** Qwen3 pairs dimension *i* with *i + D/2* (NeoX style). The other
  convention still runs and gives plausible numbers. `tests/test_rope.py` checks against
  Hugging Face's implementation.
- **Where to capture keys.** Qwen3 normalises keys before RoPE, so the hook sits on `k_norm`.
  `tests/test_identity.py` maps a model onto itself and must reproduce its logits (measured
  error 2.9e−05).

Any two dense Qwen3 models can be paired: they all have 8 KV heads of dimension 128, the same
RoPE base and the same tokenizer.

---

## Layout

```
src/kvtransfer/   rope, capture, kvstore (KV to disk once), batched_gram + stream (Gram
                  accumulators), ridge, select (top-k), mapper, graft, eval/, bench
scripts/
  02_probe_r2.py           probe: R² for every layer pair
  03_fit.py                fit mappers at each k (--all-heads, --holdout-shards)
  04_report_probe.py       probe → text report
  07_eval_tasks.py         benchmark retention
  08_paired_test.py        paired significance test
  09_writeup_figures.py    all figures
  10_speedup.py            transfer vs re-prefill latency
  11_weight_similarity.py  k_proj weight cosine between two models (Fig 7)
  run_laptop.sh            the laptop trial
  run_pair.sh              one pair on a GPU, start to finish
configs/          one YAML per pair: models, calibration, device, λ, k, tasks
tests/            fast unit tests, plus slow end-to-end checks (-m slow)
```
