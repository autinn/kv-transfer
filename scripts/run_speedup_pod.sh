#!/usr/bin/env bash
# Transfer vs re-prefill latency on the paper's Qwen3 pairs, on one rented 80 GB GPU:
#   14B -> 32B at k=8 (paper Table 5: 4-25x), then 8B -> 32B at k=12 (Table 16:
#   4.3-17.8x) reusing the 32B re-prefill timings. Paper protocol: 64-32K tokens,
#   50 warmup / 30 timed runs per cell.
#
#   bash scripts/run_speedup_pod.sh            # venv active, HF_HOME set (GPU.md)
#
# Timings are only meaningful on a GPU nothing else is using, so this refuses to start
# if any process holds the GPU, and logs every GPU process once a minute to
# <log dir>/gpu_procs.log so a run shared with another job can be spotted afterwards.
# Each result JSON is rewritten after every timed cell; a cut-off run keeps its cells.
set -euo pipefail
LOG=${LOG_DIR:-results/speedup_pod}
mkdir -p "$LOG"
step() { echo; echo "=== $(date "+%F %T") $* ==="; }

busy=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
if [ -n "$busy" ]; then
  echo "GPU is in use, not starting (timings would be wrong):"; echo "$busy"; exit 1
fi
( while true; do
    echo "$(date "+%T") $(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader | tr '\n' ';')"
    sleep 60
  done ) > "$LOG/gpu_procs.log" 2>&1 &
MON=$!
trap 'kill $MON 2>/dev/null || true' EXIT

step "models (download if missing)"
for m in Qwen/Qwen3-14B Qwen/Qwen3-32B Qwen/Qwen3-8B; do hf download "$m" > /dev/null; done

step "14B -> 32B, k=8 all-heads"
python scripts/10_speedup.py --config configs/qwen3_14b_to_32b_replication.yaml \
  --device cuda --k 8

step "8B -> 32B, k=12 all-heads, re-prefill reused from 14B -> 32B"
prev=$(ls results/qwen3_14b_to_32b/speedup_*.json | head -1)
python scripts/10_speedup.py --config configs/qwen3_8b_to_32b_speedup.yaml \
  --device cuda --k 12 --prefill-from "$prev"

step "SPEEDUP_DONE"
