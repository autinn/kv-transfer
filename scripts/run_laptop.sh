#!/usr/bin/env bash
# Laptop trial (configs/laptop_0.6b_to_1.7b.yaml): fast tests -> probe -> fit one
# all-heads k=1 mapper -> score it on 200 HellaSwag questions. About 5 GB of downloads.
#
#   bash scripts/run_laptop.sh              # Apple silicon (MPS)
#   DEVICE=cpu bash scripts/run_laptop.sh   # any machine
#
# Pass: oracle matches standalone (the transfer path is exact) and the mapper beats
# zero-copy (the fitted map carries information).
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=configs/laptop_0.6b_to_1.7b.yaml
DEVICE="${DEVICE:-mps}"
OUT=results/laptop_0.6b_to_1.7b
export HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
step() { echo; echo "=== $(date +%T) $* ==="; }
T0=$SECONDS

step "fast tests"
python -m pytest tests/ -q -m "not slow"

step "probe: capture both models' KV, R² for every layer pair"
python scripts/02_probe_r2.py --config $CONFIG --device "$DEVICE"
python scripts/04_report_probe.py $OUT/probe_r2_n64.pt | sed -n '/headline/,/^$/p'

step "fit: all-heads k=1 mapper"
python scripts/03_fit.py --config $CONFIG --device "$DEVICE" --k 1 --all-heads

step "HellaSwag, 200 questions: standalone, oracle, zero-copy, mapper"
python scripts/07_eval_tasks.py --config $CONFIG --device "$DEVICE" --tasks hellaswag \
  --only standalone oracle zero_copy mapper_k1_lam0.01_allheads

step "result"
python - <<PY
import json
d = json.load(open("$OUT/eval_tasks.json"))
for conf, r in d.items():
    h = r["hellaswag"]
    print(f"  {conf:28s} acc_norm {h['acc_norm']:.3f}   retention {h['retention_pct']:5.1f}%")
PY
echo; echo "total: $(( (SECONDS - T0) / 60 )) min"
