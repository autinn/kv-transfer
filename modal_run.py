"""Run the --all-heads refit + eval check on Modal, scoped to the CHEAP small pair only.

Deliberately does NOT run the 14B -> 32B check -- that's a separate, much more
expensive step you'd invoke only after looking at this result. Nothing here auto-chains
into it.

Setup (free, your terminal):
    uv pip install modal
    modal setup                     # links your Modal account via browser

Cost estimate printed before any GPU work starts. Default GPU is L4 (~$0.80/hr on
Modal), and this job is minutes, not hours, since it reuses the KV store already
captured and validated in REPORT3 -- no recapture, no fresh model download of anything
you haven't already pulled once this session pulls it into the Modal volume.

Run it (deployed, not `modal run --detach`):
    modal deploy modal_run.py       # registers the app; no GPU work happens yet
    modal run modal_run.py::spawn_check   # fires the job and returns immediately

`modal run --detach` was tried first and twice failed to survive independent of the
local CLI session (stalled at 0 tasks, or was cancelled mid-run) despite the pipeline
itself running correctly when it did execute. `spawn_check` looks the function up on
the *deployed* app (`Function.from_name`) and spawns it there, which genuinely
detaches on Modal's side -- spawning the local handle instead would target `modal
run`'s own ephemeral app, which dies when the entrypoint returns. The call returns a
call ID immediately and the function keeps running on Modal's infrastructure with no
further dependency on this terminal, this SSH session, or this tool call.

Check on it with:
    modal container list                         # find the running ta-... id
    modal container logs <ta-...>                # app logs may show only the image build

This prints the fit's mean R^2 (K and V) for --all-heads at k=1 and the eval_tasks
retention table on HellaSwag, then exits. Has a hard `timeout` below so a stuck run
can't bill indefinitely.
"""

import modal

app = modal.App("kv-transfer-all-heads-check")

# Mirrors pyproject.toml's [project.dependencies] plus pytest for the guardrail.
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
    .add_local_dir(
        ".", remote_path="/repo", copy=True,
        # copy=True bakes the tree into the image at build time instead of a live
        # mount; a live mount is sensitive to local files changing mid-build, which is
        # exactly what happened here (.git/info/exclude got touched mid-run and Modal
        # aborted the build as a safety check). Also excludes the local venv (~1GB,
        # irrelevant -- the image installs its own deps) and generated local results.
        ignore=[".git", ".venv", "results", "logs", "__pycache__", "*.pyc", ".DS_Store"],
    )
)

# Persists the KV store and HF model cache across runs, so a second invocation (e.g.
# after a code tweak) does not re-download Qwen3-0.6B/1.7B or recapture KV.
volume = modal.Volume.from_name("kv-transfer-cache", create_if_missing=True)

GPU = "L4"            # cheapest tier that comfortably fits this pair; see README/GPU.md
# First run has an empty cache volume: model download (~5GB) + KV capture (both models,
# 500 seqs) + all-heads fit (wider Gram than same-head) + HellaSwag eval on 4 configs,
# on a slower card than REPORT3/REPORT4 used. 45 min proved too short even on a warm
# volume: the 2026-10-06 run spent ~10 min on probe + fit, then 7 min standalone and
# 5.5 min oracle, and hit the limit at 9000/10019 contexts of zero_copy (config 3 of
# 4). 90 min fits the full pipeline with headroom and is still a hard cap (~$1.20
# on an L4; billing is for time actually used, not the cap).
TIMEOUT_S = 90 * 60


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/cache": volume},
    timeout=TIMEOUT_S,
)
def run_all_heads_check(eval_only: bool = False, k: int = 1):
    """eval_only=True skips probe + fit and resumes the eval: 07_eval_tasks.py skips
    configs already in results/eval_tasks.json and reuses the HellaSwag source KV store
    left on the volume, so only the unfinished configs are scored.

    `k` is the number of source layers per target layer for the --all-heads fit. The
    probe is skipped when its output is already on the volume (03_fit.py only reads its
    top-k selection, which does not depend on k)."""
    import subprocess
    import sys
    import time

    env = {
        "HF_HUB_DISABLE_XET": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HOME": "/cache/hf",
        "PYTHONPATH": "/repo/src",
    }

    def run(*args, **kw):
        print(f"\n$ {' '.join(args)}", flush=True)
        t0 = time.time()
        r = subprocess.run(args, cwd="/repo", env={**__import__("os").environ, **env}, **kw)
        print(f"  ({time.time()-t0:.0f}s, exit {r.returncode})", flush=True)
        if r.returncode != 0:
            raise RuntimeError(f"{args[0]} failed with exit {r.returncode}")

    import os
    os.makedirs("/cache/hf", exist_ok=True)
    os.makedirs("/cache/results/kv", exist_ok=True)
    if not os.path.islink("/repo/results") and not os.path.exists("/repo/results"):
        os.symlink("/cache/results", "/repo/results")

    print("=== guardrails (fast tests; catch a broken checkout before any GPU time) ===")
    run(sys.executable, "-m", "pytest", "tests/", "-q", "-m", "not slow")

    cfg = "configs/qwen3_0.6b_to_1.7b.a10.yaml"
    # k=1 first: same feature width (1,024 = 8 heads x 128 dim) as the same-head k=8 case
    # that already ran successfully in minutes (REPORT3/REPORT4), so this is the cheap
    # checkpoint -- does all_heads help at all within one layer -- before paying for the
    # k=8 all_heads fit. That fit was measured at ~12 min/target-layer x 28 layers before
    # TopKGrams shared one X^T X across target heads (it had kept 8 identical copies);
    # it now needs ~1.2 GB/target layer and is fp64-bound, so run it on an A100.
    K = str(k)
    if eval_only:
        print("\n=== eval_only: skipping probe + fit (mappers already on the volume) ===")
    elif os.path.exists("/repo/results/probe_r2_n500.pt"):
        print("\n=== probe: results/probe_r2_n500.pt already on the volume, skipping ===")
    else:
        print(f"\n=== probe: Stage A linear-structure R^2, config={cfg} ===")
        print("(reuses the KV store in /cache/results/kv if present -- no recapture; "
              "required before 03_fit.py, which reads this probe's top-k selection)")
        run(sys.executable, "scripts/02_probe_r2.py", "--config", cfg, "--device", "cuda")

    if not eval_only:
        print(f"\n=== fit: --all-heads, k={K}, config={cfg} ===")
        run(sys.executable, "scripts/03_fit.py", "--config", cfg, "--device", "cuda",
            "--k", K, "--all-heads")

    # The a10 config's eval.tasks lists all 5 paper benchmarks, including GSM8K
    # (generation-based, by far the slowest -- REPORT3/REPORT4 both limited themselves
    # to HellaSwag alone to control cost on faster hardware than this L4). --tasks
    # restricts to HellaSwag only, matching REPORT3/REPORT4's methodology so the
    # result is directly comparable to their numbers.
    print(f"\n=== eval: HellaSwag only, standalone / oracle / zero_copy / mapper_k{K}_lam0.01_allheads "
          "(configs already in results/eval_tasks.json are skipped) ===")
    run(sys.executable, "scripts/07_eval_tasks.py", "--config", cfg, "--device", "cuda",
        "--tasks", "hellaswag",
        "--only", "standalone", "oracle", "zero_copy", f"mapper_k{K}_lam0.01_allheads")

    volume.commit()
    print("\n=== done. KV store and results cached in the kv-transfer-cache volume. ===")


@app.local_entrypoint()
def spawn_check(eval_only: bool = False, k: int = 1, gpu: str = GPU):
    """Fire-and-forget against the DEPLOYED app (`modal deploy modal_run.py` first).

    The function is looked up by name on the deployed app rather than spawned via
    `run_all_heads_check.spawn()`: `modal run` always creates its own ephemeral app,
    so spawning the local handle targets that ephemeral app, which stops as soon as
    this entrypoint returns and takes the spawned call down with it (terminal
    "errored" status, empty message, no logs past the image build). A call on the
    deployed app has no tie to this process once spawned. Poll status with the call
    ID, or follow the container with `modal container logs <ta-...>`.

    `modal run modal_run.py::spawn_check --eval-only` resumes an interrupted eval
    without redoing the probe and fit. `--k 8 --gpu A100-40GB` runs the k=8 all-heads
    fit on an A100: the fit is fp64-Gram-bound, and the L4's fp64 rate (~0.5 TFLOPS)
    would make it ~40-60 min against a few minutes on an A100. The GPU is applied at
    spawn time with `with_options`, so no redeploy is needed to switch cards.
    """
    print(f"GPU: {gpu}  |  k={k}  |  hard timeout: {TIMEOUT_S/60:.0f} min  |  Modal's own {gpu} rate applies")
    print("This runs ONLY the small-pair (0.6B->1.7B) --all-heads check.")
    print("It does NOT run the 14B->32B comparison -- that's a separate, bigger job.")
    if eval_only:
        print("eval_only: skipping probe + fit, resuming the HellaSwag eval.")
    print()
    deployed = modal.Function.from_name(app.name, "run_all_heads_check")
    if gpu != GPU:
        deployed = deployed.with_options(gpu=gpu)
    call = deployed.spawn(eval_only=eval_only, k=k)
    print(f"Spawned, call id: {call.object_id}")
    print("This process can now exit; the job keeps running on Modal regardless.")
    print("Check progress with: modal container list, then modal container logs <ta-...>")


@app.local_entrypoint()
def main():
    """`modal run modal_run.py` (no --detach): runs in the foreground, blocks until
    done. Kept for a quick local test; prefer `spawn_check` for anything you don't
    want tied to this terminal session."""
    print(f"GPU: {GPU}  |  hard timeout: {TIMEOUT_S/60:.0f} min  |  Modal's own {GPU} rate applies")
    run_all_heads_check.remote()
