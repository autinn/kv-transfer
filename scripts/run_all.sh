#!/usr/bin/env bash
# Full pipeline, start to finish. Written for a rented GPU box but runs anywhere.
#
#   bash scripts/run_all.sh              # paper-faithful: 500 calibration sequences
#   N_SEQS=100 bash scripts/run_all.sh   # quicker pass
#   DEVICE=cpu bash scripts/run_all.sh   # no GPU
#   CONFIG=configs/qwen3_14b_to_32b.yaml bash scripts/run_all.sh   # the paper's pair
#
# On an A10/L4 the whole thing is roughly 30-40 minutes including downloads.

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/qwen3_0.6b_to_1.7b.yaml}"
N_SEQS="${N_SEQS:-500}"
DEVICE="${DEVICE:-cuda}"
KS="${KS:-1 2 4 8 12}"
OUT="$(python -c "import yaml;print(yaml.safe_load(open('${CONFIG}')).get('output_dir','results'))")"

export HF_HUB_DISABLE_XET=1        # Xet keeps a second copy of every blob
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

echo "=== environment ==="
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

echo
echo "=== 0. guardrails (must pass before any result is trustworthy) ==="
pytest tests/ -q -m "not slow"
pytest tests/ -q -m slow

echo
echo "=== 1-2. linear-structure probe (n=${N_SEQS}) ==="
python -u scripts/02_probe_r2.py --config "${CONFIG}" \
  --n-seqs "${N_SEQS}" --device "${DEVICE}"

echo
echo "=== report ==="
python -u scripts/04_report_probe.py "${OUT}/probe_r2_n${N_SEQS}.pt" \
  | tee "${OUT}/probe_report_n${N_SEQS}.txt"

echo
echo "=== 3. fit production mappers (k = ${KS}) ==="
python -u scripts/03_fit.py --config "${CONFIG}" --device "${DEVICE}" \
  --n-seqs "${N_SEQS}" --k ${KS} --identity-baseline

echo
echo "=== 4. prefix-conditioned perplexity vs all baselines ==="
python -u scripts/05_eval.py --config "${CONFIG}" --device "${DEVICE}"

echo
echo "=== 5. benchmark retention (eval.tasks in the config) ==="
python -u scripts/07_eval_tasks.py --config "${CONFIG}" --device "${DEVICE}"

echo
echo "=== done. ${OUT}/ ==="
ls -la "${OUT}/"
