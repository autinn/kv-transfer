# Next Steps — Running This On A Rented GPU

Everything is implemented and tested. What remains is one compute run.

**Short answer: rent a single 24 GB GPU for about an hour.** One GPU, not several —
nothing here is distributed, and the bottleneck is sequential model forwards.

---

## 1. Which GPU, and how many

### How many: **one**

The pipeline loads one model at a time and streams sequences through it. There is no
data-parallel or model-parallel component, and adding GPUs would not help: the second
would sit idle. Renting two is pure waste.

### Which one

Required: **≥8 GB VRAM** and **≥16 GB host RAM**. The measured requirements:

| | VRAM peak | host RAM |
|---|---|---|
| Qwen3-1.7B weights + activations (measured, REPORT.md §5) | ~3.4 GB | — |
| statistics (full 28×28×8 grid, fp32) | — | 0.88 GB |
| captured KV at the shipped `chunk: 5` (`configs/qwen3_0.6b_to_1.7b.yaml`) | — | ~0.9 GB |
| **total** | **~3.4 GB** | **~1.8 GB** |

Host RAM is not additive with VRAM — the model lives on the GPU, the statistics and
captured KV live on the host — and the config's own comment
(`configs/qwen3_0.6b_to_1.7b.yaml:14-16`) notes peak RSS is `max(...)` across stages,
not a sum, so these are conservative upper bounds. Raising `runtime.chunk` in the config
trades host RAM for fewer model reloads; the shipped default favors low memory.

So even a 16 GB card is roomy. Recommendations in order of value:

| GPU | VRAM | ~$/hr | Verdict |
|---|---|---|---|
| **L4** | 24 GB | $0.40–0.80 | **Best value.** Ample headroom, cheap, widely available |
| **A10 / A10G** | 24 GB | $0.60–1.00 | Equally good; often the default on AWS/Lambda |
| RTX 4090 | 24 GB | $0.35–0.70 | Cheapest on Vast.ai / RunPod; fine for this |
| A100 40 GB | 40 GB | $1.20–2.00 | Overkill — you will not use the extra VRAM or bandwidth |
| T4 | 16 GB | $0.20–0.40 | Works, but ~2–3× slower; fine if you are optimizing for cost |

**Pick an L4 or a 4090.** Avoid A100/H100 — this workload is small and sequential, so
you would pay 3× for maybe 1.5× the speed.

Providers: Lambda Labs and RunPod are simplest; Vast.ai is cheapest if you do not mind
variable hosts. Any of them will do — you need the box for under an hour.

### Budget

| Run | GPU time | Cost |
|---|---|---|
| Full pipeline, n=500 (paper-faithful) | ~30–40 min (see README "Expected runtime") | **~$0.30–0.70** |
| Plus setup and model downloads | ~15 min | included |
| Optional: add the 1.7B→4B pair | +40 min | +~$0.50 |

**Budget $2 and you are comfortably covered**, including a re-run if something needs
adjusting.

---

## 2. Setup on the box

```bash
git clone <this repo> && cd kv-transfer

curl -LsSf https://astral.sh/uv/install.sh | sh    # or use the system python
uv venv --python 3.12 && uv pip install -e ".[dev]"

source .env.sh                     # HF_HUB_DISABLE_XET etc. — see .env.sh for the
                                    # up-to-date list; do not hardcode it here
export TOKENIZERS_PARALLELISM=false  # silences a tokenizer fork warning under multiprocessing
```

Then, before anything else:

```bash
pytest tests/ -q -m "not slow"   # 20 tests, ~15 s
pytest tests/ -q -m slow         # 4 guardrails, downloads Qwen3-0.6B
```

**If the guardrails fail, stop.** They verify that grafting a cache reproduces the
model's logits; if that is broken, every number downstream is meaningless. On this
machine they pass at 2.9e-05.

---

## 3. The run

```bash
bash scripts/run_all.sh          # N_SEQS=500 DEVICE=cuda by default
```

That executes, in order: guardrails → R² probe → report → mapper fits → perplexity eval.
It stops on the first failure.

If you would rather go incrementally (recommended the first time — the probe result
determines whether the rest is worth running):

```bash
# ~10 min: the decisive experiment
python scripts/02_probe_r2.py --n-seqs 100 --device cuda --out results/probe_n100.pt
python scripts/04_report_probe.py results/probe_n100.pt
```

**Read that output before continuing.** See §5 for how to interpret it.

---

## 4. What to test, in priority order

### Priority 1 — the linear-structure probe (the whole question)

```bash
python scripts/02_probe_r2.py --n-seqs 500 --device cuda --out results/probe_n500.pt
python scripts/04_report_probe.py results/probe_n500.pt | tee results/probe_report.txt
```

Produces the 28×28 R² grid for keys (content space), values, and the layer-aligned
post-RoPE comparison. **This is the paper's central evidence and the input to everything
else.** ~30–40 min at n=500, ~10 min at n=100 (see README "Expected runtime").

### Priority 2 — production fits and the k-sweep

```bash
python scripts/03_fit.py --probe results/probe_n500.pt --n-seqs 500 \
                        --k 1 2 4 8 12 16 --identity-baseline
```

The paper's key finding is that multiple source layers beat one (R² 0.56 → 0.79 for
keys). The `--identity-baseline` flag additionally fits layer *l* → layer *l*. Because
this pair has equal depth — unlike any pair in the paper — that baseline is a fair
comparison, and **if top-k cannot beat it, cross-layer selection is not earning its
complexity here.** Worth reporting either way.

### Priority 3 — retention

```bash
python scripts/05_eval.py --out results/eval_ppl.json
```

Prefix-conditioned perplexity on WikiText-2 (a different domain from the FineWeb-Edu
calibration, so this measures transfer rather than memorization), against four
baselines:

- **target standalone** — the ceiling
- **oracle graft** — target's own cache through the graft path; must match the ceiling,
  otherwise the harness is at fault rather than the mapper
- **zero-copy** — source cache pasted in unmapped; answers *is the mapper doing anything
  at all?*
- **mean prediction** — per-channel means, the R² = 0 floor

**Do not report a retention number without the zero-copy control.** If a fitted mapper
barely beats pasting the cache in unchanged, the models were simply already similar and
the regression learned nothing.

### Priority 4 — length generalization

```bash
python scripts/05_eval.py --prefix-len 4096 --out results/eval_ppl_4k.json
```

Mappers are fitted at positions 0–1023. This checks they still work at 4096, which is
the actual justification for fitting in content space. Cheap, and it tests a claim the
paper makes but this pair otherwise cannot verify (see the caveat in REPORT.md §6).

### Priority 5 — optional extras, if the above is positive

- **Second pair, 1.7B → 4B.** Also matched-KV (8 × 128) but crosses a depth boundary
  (28 → 36 layers), so it exercises the cross-layer machinery properly. Needs ~9 GB
  VRAM — fine on a 24 GB card. Write a config mirroring the existing one.
- **λ sweep.** Nearly free: one Cholesky per λ on a cached Gram. `configs/*.yaml`
  already lists `[0.001, 0.01, 0.1]`.
- **Latency benchmark.** Transfer vs full re-prefill at 1k/4k/16k tokens — the paper's
  2.7–25× speedup claim. Not yet implemented; the harness is there if you want it.
- **Calibration-domain sensitivity.** The paper tests Wikipedia and CodeAlpaca in
  addition to FineWeb-Edu. `data.py` is structured to make adding one easy.

---

## 5. How to read the result

The number that matters first: **best single-source K R²**, printed by
`04_report_probe.py`.

| Result | Interpretation | What to do |
|---|---|---|
| **≥ 0.7** | Comparable to the paper's 0.81 on 14B→32B. The method holds at sub-2B scale. | Run everything. This is a clean positive replication. |
| **0.4 – 0.7** | Partial structure — weaker than 3B+ pairs but clearly non-trivial. | Run the k-sweep; multi-source selection may recover much of the gap. This is the most interesting outcome scientifically. |
| **< 0.4** | The two models do not share enough linear structure at this scale. | **Still run Priority 2 briefly** to confirm multi-source does not rescue it, then stop. Report the negative. |

**A negative result is a real finding, not a failed experiment.** The paper's smallest
pair is 3B and nothing below that has been published. "The closed-form linear mapping
does not transfer below ~3B" is a genuine contribution about the scale-dependence of the
claim, and this repo was built to measure that rather than to confirm the paper.

Also worth noting from the probe output:

- **Is the best source layer usually the aligned one?** The report prints how many
  target layers pick their own index. Heavy off-diagonal structure would be a finding in
  itself.
- **Does RoPE stripping help?** The report compares `K_content` against `K_rope` on the
  diagonal. For this pair `R_s == R_t`, so a large gap would be surprising and worth
  investigating.
- **Keys vs values.** The paper finds keys more predictable than values (0.79 vs 0.65).
  Check whether that ordering reproduces.

---

## 6. Getting results back

Everything lands in `results/` and is small (a few MB):

```bash
tar czf kv-results.tgz results/
# then scp / rsync it off the box before you terminate the instance
```

Keep `results/probe_*.pt` — it is the expensive artifact, and the k-sweep and all
reporting can be re-derived from it without re-running the models.

---

## 7. Cost-saving notes

- **Run at n=100 first** (~10 min, ~$0.15). It answers the central question. Only scale
  to n=500 if the answer is promising — n=100 already gives 25,600 observations per head
  against a 128-feature fit.
- **Terminate the instance when done.** Obvious, but the usual way these bills grow.
- **Use a persistent volume or `scp` results off** before terminating; re-running to
  recover lost output is the expensive mistake.
- Model downloads are ~5 GB and take a few minutes; that time is billed, so do not
  restart the box unnecessarily.

---

## Validating against the paper: Qwen3-14B → Qwen3-32B

The paper reports detailed numbers for this pair, so running it checks the
implementation directly. `configs/qwen3_14b_to_32b.yaml` is set up for it.

**Hardware: one 80 GB GPU** (H100 or A100-80GB). Qwen3-32B in bf16 is ~64 GB; the two
models are never resident together. The L4 above is too small. Disk: ~55 GB for the
calibration KV stores plus `eval.store_budget_gb` (60 GB) for benchmark rounds. Put
`runtime.store_dir` on the instance's largest volume.

```bash
CONFIG=configs/qwen3_14b_to_32b.yaml bash scripts/run_all.sh
```

**Cheap check (stages 1–3, no benchmarks)**: compare the probe and fit output to the
paper's reported R² for this pair:

| | paper |
|---|---|
| best single-source key R² (best cell) | 0.81 |
| k = 1 mean R², keys / values | 0.56 / 0.32 |
| k = 8 mean R², keys / values | 0.79 / 0.65 |
| k = all mean R², keys / values | 0.85 / 0.76 |

**Full check**: `07_eval_tasks.py` retention on the five benchmarks against the paper's
97.6% average at k = 8 (HellaSwag 80.70 absolute at k = 8, 44.81 at k = 1).

Two known differences from the paper:

1. **Features per target head — confirmed against the paper's text, now implemented.**
   The default still maps target head *h* from the same head *h* in each selected
   source layer (k·128 inputs). The paper's method section states the feature vector is
   "the concatenation of all Key heads from source layer *l_i*," i.e. k·8·128 inputs,
   explicitly "enabling cross-head information flow" — matching its quoted mapper sizes
   (3.36B parameters for Llama 8B→70B at k = 20). This is now available as
   `--all-heads` on `scripts/03_fit.py` (see `stream.TopKGrams` and `mapper.KVMapper`,
   both take `all_heads=True`). REPORT3 and REPORT4 both found the same shortfall at
   two very different scales (k=1→k=8 R² gain of +0.11–0.12 here vs. the paper's +0.23),
   which rules out scale as the explanation and makes this the leading candidate fix —
   not yet run; refit at k=8 with `--all-heads` on the already-captured 14B→32B store
   and compare against REPORT4's numbers.
2. **λ scaling.** `ridge.py` multiplies λ by the mean feature variance, so λ = 0.01 here
   is not the paper's λ = 0.01. Less likely to explain the multi-layer gap now that this
   repo's k=1 R² already matches the paper's closely under the current scaling.
