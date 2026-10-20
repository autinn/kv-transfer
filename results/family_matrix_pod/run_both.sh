set -uo pipefail
cd /root/kv-transfer
source /root/venv/bin/activate
export HF_HOME=/root/hf HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src
step() { echo; echo "=== $(date "+%F %T") $* ==="; }
step "guardrails (fast tests)"
python -m pytest tests -q -m "not slow" 2>&1 | tail -3 || { echo "TESTS_FAILED"; exit 1; }
step "disk before pair 1"; df -h / | tail -1
bash scripts/run_pair.sh configs/qwen3_4b_to_8b.yaml || { echo "PAIR1_FAILED"; exit 1; }
step "free 8B pieces before pair 2"
rm -rf results/kv/Qwen__Qwen3-8B /root/hf/hub/models--Qwen--Qwen3-8B
df -h / | tail -1
bash scripts/run_pair.sh configs/qwen3_1.7b_to_4b.yaml || { echo "PAIR2_FAILED"; exit 1; }
step "disk at end"; df -h / | tail -1
echo "ALL_DONE"
