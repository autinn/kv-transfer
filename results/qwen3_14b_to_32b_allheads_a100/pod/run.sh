set -euo pipefail
cd /workspace/kv-transfer
source /workspace/venv/bin/activate
export HF_HOME=/workspace/hf HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src
CFG=configs/repl_pod.yaml
step() { echo; echo "=== $(date "+%F %T") $* ==="; }
# Weights download in parallel with the probe; the probe waits on files (GPU.md section 5).
step "download both models (background)"
( hf download Qwen/Qwen3-14B > /root/logs/dl_14b.log 2>&1; echo "14B done $(date +%T)" >> /root/logs/dl_14b.log ) &
( hf download Qwen/Qwen3-32B > /root/logs/dl_32b.log 2>&1; echo "32B done $(date +%T)" >> /root/logs/dl_32b.log ) &
wait
step "probe (Stage A) + calibration capture, both models"
python scripts/02_probe_r2.py --config $CFG --device cuda
step "fit k=8 same-head"
python scripts/03_fit.py --config $CFG --device cuda --k 8
step "fit k=8 all-heads"
python scripts/03_fit.py --config $CFG --device cuda --k 8 --all-heads
step "eval HellaSwag: same-head k8, all-heads k8 (baselines from REPORT4)"
python scripts/07_eval_tasks.py --config $CFG --device cuda --tasks hellaswag \
  --only mapper_k8_lam0.01 mapper_k8_lam0.01_allheads
step "paired test"
python scripts/08_paired_test.py --config $CFG
step "RUN_DONE"
