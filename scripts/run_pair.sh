#!/usr/bin/env bash
# One matched-KV pair, the 14B->32B replication protocol, on a rented GPU box:
#   probe (captures both calibration stores) -> all-heads k=1 and k=8 fits ->
#   held-out R^2 refits (last 2 shards) -> HellaSwag on baselines + both mappers ->
#   paired test. Resumable: every stage reuses its outputs if present.
#
#   bash scripts/run_pair.sh configs/qwen3_4b_to_8b.yaml
#   bash scripts/run_pair.sh configs/qwen3_1.7b_to_4b.yaml
#
# Assumes the venv is active and HF_HOME / PYTHONPATH are set (see GPU.md).
#
# Disk plan for a 120 GB box (A100-80GB pod, 2026-10-07). Peak during 4B->8B:
#   weights 4B+8B ~24 GB, calibration stores 2 x 19 GB, venv ~6 GB, mappers ~7 GB,
#   one HellaSwag round store <=30 GB (eval.store_budget_gb)  ->  ~105 GB.
# keep_stores: false deletes the round store after a clean eval. Before the second
# pair, free the 8B pieces it does not need (~35 GB):
#   rm -rf results/kv/Qwen__Qwen3-8B && hf cache delete --repo model/Qwen/Qwen3-8B  (or rm its hub dir)
# The 4B calibration store is shared with 1.7B->4B and must stay.
set -euo pipefail
CFG=${1:?usage: run_pair.sh <config.yaml>}
OUT=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "$CFG")
step() { echo; echo "=== $(date "+%F %T") $* ==="; }

step "models (download if missing)"
for m in $(python -c "import yaml,sys; c=yaml.safe_load(open(sys.argv[1])); print(c['source'], c['target'])" "$CFG"); do
  hf download "$m" > /dev/null
done

step "probe (Stage A) + calibration capture, both models  [$CFG]"
python scripts/02_probe_r2.py --config "$CFG" --device cuda
python scripts/04_report_probe.py "$OUT"/probe_r2_n500.pt | tee "$OUT"/probe_report_n500.txt

step "fit all-heads k=1 and k=8"
python scripts/03_fit.py --config "$CFG" --device cuda --k 1 8 --all-heads

step "held-out R^2 (fit on 18 of 20 shards, score on the last 2)"
python scripts/03_fit.py --config "$CFG" --device cuda --k 1 8 --all-heads \
  --holdout-shards 2 --outdir "$OUT"/mappers_heldout

step "HellaSwag: standalone, oracle, zero-copy, all-heads k=1, all-heads k=8"
python scripts/07_eval_tasks.py --config "$CFG" --device cuda --tasks hellaswag \
  --mappers "$OUT"/mappers/mapper_k1_lam0.01_allheads.pt "$OUT"/mappers/mapper_k8_lam0.01_allheads.pt

step "paired test"
python scripts/08_paired_test.py --config "$CFG" || true

step "PAIR_DONE $CFG"
