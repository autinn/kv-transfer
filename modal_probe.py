"""Run the linear-structure probe (scripts/02_probe_r2.py) for one pair on a Modal H100.

Probe only: captures both models' calibration KV, scores every (source layer, target
layer, head) cell, writes <output_dir>/probe_r2_n500.pt and the text report. No fit, no
benchmark. Built for 8B -> 32B (configs/qwen3_8b_to_32b.yaml); 14B -> 32B took 12.6 min
for the same step on an H100 PCIe (REPORT4.md section 3).

    MODAL_PROFILE=autinn modal deploy modal_probe.py
    MODAL_PROFILE=autinn modal run modal_probe.py::spawn
    MODAL_PROFILE=autinn modal volume get kv-transfer-cache results/qwen3_8b_to_32b results/

Weights are cached on the kv-transfer-cache volume (hf/), so a later full run of the pair
skips the ~80 GB download. The calibration stores (~55 GB) go to the container's local
disk and are discarded; recapturing them takes ~7 min.
"""

import modal

app = modal.App("kv-transfer-probe")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.51",
        "accelerate>=0.26",
        "datasets>=2.20",
        "safetensors>=0.4",
        "pyyaml>=6.0",
        "numpy>=1.26",
        "tqdm>=4.66",
        "pytest>=8.0",
    )
    .add_local_dir(".", remote_path="/repo", copy=True,
                   ignore=[".git", ".venv", "results", "logs", "__pycache__", "*.pyc", ".DS_Store"])
)
volume = modal.Volume.from_name("kv-transfer-cache", create_if_missing=True)

GPU = "H100"
TIMEOUT_S = 60 * 60   # hard cap; the probe itself should take ~15-25 min incl. downloads


@app.function(image=image, gpu=GPU, volumes={"/cache": volume}, timeout=TIMEOUT_S)
def run_probe(config: str = "configs/qwen3_8b_to_32b.yaml"):
    import os
    import subprocess
    import sys
    import time

    import yaml

    out = yaml.safe_load(open(f"/repo/{config}"))["output_dir"]   # e.g. results/qwen3_8b_to_32b
    env = {**os.environ, "HF_HOME": "/cache/hf", "PYTHONPATH": "/repo/src",
           "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"}

    def run(*args):
        print(f"\n$ {' '.join(args)}", flush=True)
        t0 = time.time()
        r = subprocess.run(args, cwd="/repo", env=env)
        print(f"  ({(time.time() - t0) / 60:.1f} min, exit {r.returncode})", flush=True)
        if r.returncode != 0:
            raise RuntimeError(f"{args[1]} failed with exit {r.returncode}")

    # results/ is local disk (KV stores land in results/kv); only the pair's output
    # directory points at the volume.
    os.makedirs("/repo/results", exist_ok=True)
    os.makedirs(f"/cache/{out}", exist_ok=True)
    if not os.path.exists(f"/repo/{out}"):
        os.symlink(f"/cache/{out}", f"/repo/{out}")

    run(sys.executable, "-m", "pytest", "tests/", "-q", "-m", "not slow")
    try:
        run(sys.executable, "scripts/02_probe_r2.py", "--config", config, "--device", "cuda")
        report = subprocess.run([sys.executable, "scripts/04_report_probe.py", f"{out}/probe_r2_n500.pt"],
                                cwd="/repo", env=env, capture_output=True, text=True)
        print(report.stdout, report.stderr)
        open(f"/repo/{out}/probe_report_n500.txt", "w").write(report.stdout)
    finally:
        volume.commit()
    print(f"\n=== done. Fetch with: modal volume get kv-transfer-cache {out} results/ ===")


@app.local_entrypoint()
def spawn(config: str = "configs/qwen3_8b_to_32b.yaml"):
    """Fire-and-forget on the DEPLOYED app (see modal_run.py's spawn_check for why)."""
    print(f"GPU: {GPU}  |  probe only  |  {config}  |  hard timeout: {TIMEOUT_S // 60} min")
    call = modal.Function.from_name(app.name, "run_probe").spawn(config=config)
    print(f"Spawned, call id: {call.object_id}")
    print("Follow with: modal container list, then modal container logs <ta-...>")
