"""The real scripts, end to end, on a tiny unequal-depth pair (2 -> 3 layers).

Catches wiring errors that unit tests cannot: config keys, default paths, the probe ->
fit -> eval hand-offs, store reuse between stages, and round-based task eval. Uses the
Qwen3 tokenizer from the local HF cache and the benchmark datasets (downloaded on first
use), hence `slow`.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from conftest import tiny_qwen3

pytestmark = pytest.mark.slow
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pair_dir(tmp_path_factory):
    from transformers import AutoTokenizer

    d = tmp_path_factory.mktemp("pair")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    for name, layers, seed in (("src", 2, 0), ("tgt", 3, 1)):
        m = tiny_qwen3(layers, seed, vocab=len(tok))
        m.save_pretrained(d / name)
        tok.save_pretrained(d / name)
    return d


def _config(pair_dir, out, budget_gb=50.0):
    cfg = {
        "source": str(pair_dir / "src"), "target": str(pair_dir / "tgt"),
        "output_dir": str(out),
        "calibration": {"dataset": "synthetic", "vocab": 5000, "n_seqs": 12, "seq_len": 64,
                        "stride": 2, "seed": 0},
        "runtime": {"chunk": 5, "batch_size": 2, "dtype": "float32", "device": "cpu",
                    "store_dir": str(out / "kv")},
        "probe": {"gram_dtype": "float32", "rope_diagonal_only": True},
        "fit": {"lambdas": [0.01, 0.1], "k_sweep": [1, 2], "max_gram_gb": 1e-5},
        "eval": {"dataset": "synthetic", "wikitext_seqs": 3, "seq_len": 48, "prefix_len": 24,
                 "min_continuation": 16, "store_budget_gb": budget_gb,
                 "tasks": {"arc_challenge": {"limit": 3}, "winogrande": {"limit": 2},
                           "mmlu": {"limit": 2}, "hellaswag": {"limit": 2},
                           "gsm8k": {"limit": 1, "num_fewshot": 1, "max_new_tokens": 6}}},
    }
    p = out / "cfg.yaml"
    out.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg))
    return p


def _run(*args):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


def test_full_pipeline_unequal_depth(pair_dir, tmp_path):
    out = tmp_path / "run"
    cfg = _config(pair_dir, out)

    log = _run("scripts/02_probe_r2.py", "--config", cfg)
    assert "2 layers | tgt" in log and "3 layers" in log
    probe = out / "probe_r2_n12.pt"
    assert probe.exists()

    import torch
    d = torch.load(probe)
    assert d["r2"]["k_content"].shape == (2, 3, 2)
    assert torch.isfinite(d["r2"]["k_content"]).all()
    assert torch.isfinite(__import__("kvtransfer.select", fromlist=["x"]).aligned_cells(
        d["r2"]["k_rope"].mean(-1))).all()

    log = _run("scripts/03_fit.py", "--config", cfg, "--identity-baseline")
    assert "reusing KV store" in log and "loading" not in log      # capture happened once
    assert "target-layer group(s)" in log
    mappers = sorted(p.name for p in (out / "mappers").glob("*.pt"))
    assert mappers == sorted(f"mapper_{n}_lam{l}.pt" for n in ("k1", "k2", "identity") for l in (0.01, 0.1))

    _run("scripts/04_report_probe.py", str(probe))

    log = _run("scripts/05_eval.py", "--config", cfg)
    res = json.loads((out / "eval_ppl.json").read_text())
    assert res["oracle_graft"] == pytest.approx(res["target_standalone"], rel=1e-3)
    for k in ("zero_copy", "mean_pred", "mapper_k2_lam0.01", "mapper_identity_lam0.01"):
        assert k in res and res[k] > 1

    log = _run("scripts/07_eval_tasks.py", "--config", cfg,
               "--only", "standalone", "oracle", "zero_copy", "mapper_k2_lam0.01")
    one = json.loads((out / "eval_tasks.json").read_text())
    for task, m in one["standalone"].items():
        assert one["oracle"][task] == pytest.approx(m, abs=1e-9, rel=1e-9) or task == "gsm8k"
    assert set(one) == {"standalone", "oracle", "zero_copy", "mapper_k2_lam0.01"}
    assert not list((out / "kv" ).rglob("task_*"))               # round stores deleted

    # Rerun resumes: nothing left to score, no model loads.
    log = _run("scripts/07_eval_tasks.py", "--config", cfg,
               "--only", "standalone", "oracle", "zero_copy", "mapper_k2_lam0.01")
    assert "loading" not in log

    # A tiny disk budget splits the work into many rounds; results must not change.
    out2 = tmp_path / "rounds"
    cfg2 = _config(pair_dir, out2, budget_gb=2e-5)
    (out2 / "mappers").mkdir(parents=True)
    src_m = out / "mappers" / "mapper_k2_lam0.01.pt"
    (out2 / "mappers" / src_m.name).write_bytes(src_m.read_bytes())
    log = _run("scripts/07_eval_tasks.py", "--config", cfg2, "--only", "zero_copy", "mapper_k2_lam0.01",
               "standalone")
    n_rounds = int(log.split(" round(s)")[0].split("\n")[-1])
    assert n_rounds > 3
    many = json.loads((out2 / "eval_tasks.json").read_text())
    for conf in ("standalone", "zero_copy", "mapper_k2_lam0.01"):
        for task in many[conf]:
            for metric, v in many[conf][task].items():
                if metric != "seconds":
                    assert v == pytest.approx(one[conf][task][metric], abs=1e-9), (conf, task, metric)


def test_fit_refuses_probe_from_other_pair(pair_dir, tmp_path):
    import torch

    out = tmp_path / "run"
    cfg = _config(pair_dir, out)
    torch.save({"r2": {"k_content": torch.rand(3, 3, 2), "v": torch.rand(3, 3, 2)},
                "Ls": 3, "Lt": 3, "H": 2, "D": 16, "n_seqs": 12,
                "source": "a", "target": "b"}, out / "probe_r2_n12.pt")
    r = subprocess.run([sys.executable, "scripts/03_fit.py", "--config", str(cfg)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "different model pair" in r.stderr
