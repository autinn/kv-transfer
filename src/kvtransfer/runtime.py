"""Model loading, memory release and config helpers shared by every script.

Previously each script reached into `scripts/02_probe_r2.py` for these via
`import_module`; they live here so the library does not depend on a script.
"""

from __future__ import annotations

import gc
import re
from pathlib import Path

import torch


def resolve_runtime(rt: dict, device: str | None = None) -> dict:
    """Apply a --device override and fix dtypes the device cannot run."""
    rt = dict(rt)
    if device:
        rt["device"] = device
    if rt["device"] == "cpu" and rt.get("dtype") in ("float16", "bfloat16"):
        rt["dtype"] = "float32"          # half-precision matmuls are slow or unsupported on CPU
    return rt


def load_model(name: str, dtype, device: str, device_map: str | None = None):
    """Load a causal LM straight onto its device.

    Loading with `device_map` puts weights on the GPU as they are read, so a 64 GB
    Qwen3-32B does not need 64 GB of host RAM first. `device_map="auto"` shards across
    every visible GPU for models larger than one card.
    """
    from transformers import AutoModelForCausalLM

    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    if device_map or str(device).startswith("cuda"):
        m = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map=device_map or device)
    else:
        m = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        if device != "cpu":
            m = m.to(device)
    m.eval()
    return m


def input_device(model):
    """Device the model's input embeddings live on (differs from `cuda` under sharding)."""
    return model.get_input_embeddings().weight.device


def free(*objs):
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


class LazyModel:
    """Load a model on first use, so a stage that finds everything cached never loads it."""

    def __init__(self, name: str, rt: dict):
        self.name, self.rt = name, rt
        self._model = None

    def get(self):
        if self._model is None:
            print(f"loading {self.name} ({self.rt['dtype']}, {self.rt['device']}) ...", flush=True)
            self._model = load_model(
                self.name, self.rt["dtype"], self.rt["device"], self.rt.get("device_map")
            )
        return self._model

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def release(self):
        if self._model is not None:
            m, self._model = self._model, None
            free(m)


def rope_theta(cfg) -> float:
    """rope_theta across transformers versions (`rope_parameters` in v5, attribute before)."""
    params = getattr(cfg, "rope_parameters", None)
    if isinstance(params, dict) and "rope_theta" in params:
        return float(params["rope_theta"])
    return float(cfg.rope_theta)


def kv_dims(cfg) -> tuple[int, int, int]:
    """(layers, kv_heads, head_dim) from a model config."""
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return cfg.num_hidden_layers, cfg.num_key_value_heads, head_dim


def pair_dims(src_cfg, tgt_cfg) -> tuple[int, int, int, int]:
    """(Ls, Lt, H, D) for a source/target pair, refusing pairs the method cannot map.

    Layer counts may differ (14B has 40, 32B has 64); KV head count and head dimension
    may not, since each target head is regressed on same-index source heads.
    """
    Ls, Hs, Ds = kv_dims(src_cfg)
    Lt, Ht, Dt = kv_dims(tgt_cfg)
    if (Hs, Ds) != (Ht, Dt):
        raise ValueError(
            f"not a matched-KV pair: source {Hs} kv heads x {Ds} vs target {Ht} x {Dt}"
        )
    return Ls, Lt, Hs, Ds


def slug(name: str) -> str:
    """Filesystem-safe name for a model id or path."""
    return re.sub(r"[^A-Za-z0-9._-]+", "__", name.strip("/"))


def output_dir(cfg: dict, root: Path) -> Path:
    """Where a config's results go. Defaults to `results/` for the original 0.6B->1.7B layout."""
    return root / cfg.get("output_dir", "results")


def store_dir(cfg: dict, root: Path) -> Path:
    """Where captured KV shards go. Large for big models (tens of GB), so configurable."""
    rt = cfg.get("runtime", {})
    p = Path(rt.get("store_dir", "results/kv"))
    return p if p.is_absolute() else root / p
