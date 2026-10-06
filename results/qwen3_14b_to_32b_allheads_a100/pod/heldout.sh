cd /workspace/kv-transfer && source /workspace/venv/bin/activate
export HF_HOME=/workspace/hf PYTHONPATH=src TOKENIZERS_PARALLELISM=false
for x in "" "--all-heads"; do
  python scripts/03_fit.py --config configs/repl_pod.yaml --device cuda --k 8 --holdout-shards 2 --outdir /root/ho_mappers $x || { echo "HO_EXIT 1"; exit 1; }
done
echo "HO_EXIT 0"
