"""Capture content-space (pre-RoPE) K and V from a Qwen3 model.

Hook placement is the subtle part. Qwen3's attention forward is:

    key_states   = k_norm(k_proj(h).view(shape)).transpose(1, 2)
    value_states = v_proj(h).view(shape).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)      <-- rotation happens HERE
    past_key_values.update(k, v, layer_idx)          <-- cache stores POST-RoPE k

So content space is the output of `k_norm`, before `apply_rotary_pos_emb`. Hooking the
cache instead would capture rotated keys and make the fitted map position-dependent,
which silently breaks generalization beyond the calibration length.

`k_norm` outputs (batch, seq, kv_heads, head_dim) -- the transpose to
(batch, kv_heads, seq, head_dim) happens after the module returns, so we transpose here.
`v_proj` outputs flat (batch, seq, kv_heads * head_dim) and needs reshaping.
"""

from __future__ import annotations

import torch


class KVCapture:
    """Registers forward hooks on every layer's k_norm and v_proj.

    Usage:
        with KVCapture(model) as cap:
            model(input_ids)
        k = cap.keys[layer]    # (batch, kv_heads, seq, head_dim), pre-RoPE
        v = cap.values[layer]
    """

    def __init__(self, model, layers: list[int] | None = None, store_dtype=torch.float16,
                 device="cpu"):
        self.model = model
        cfg = model.config
        self.n_layers = cfg.num_hidden_layers
        self.kv_heads = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.layers = list(range(self.n_layers)) if layers is None else sorted(layers)
        self.store_dtype = store_dtype
        self.device = device          # where captured tensors are kept
        self.keys: dict[int, torch.Tensor] = {}
        self.values: dict[int, torch.Tensor] = {}
        self._handles: list = []

    def _decoder_layers(self):
        base = getattr(self.model, "model", self.model)
        return base.layers

    def _k_hook(self, idx: int):
        def hook(_mod, _inp, out):
            # (b, seq, kv_heads, head_dim) -> (b, kv_heads, seq, head_dim)
            self.keys[idx] = out.detach().transpose(1, 2).to(self.device, self.store_dtype)
        return hook

    def _v_hook(self, idx: int):
        def hook(_mod, _inp, out):
            b, s, _ = out.shape
            v = out.detach().view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
            self.values[idx] = v.to(self.device, self.store_dtype)
        return hook

    def __enter__(self) -> "KVCapture":
        for i, layer in enumerate(self._decoder_layers()):
            if i not in self.layers:
                continue
            attn = layer.self_attn
            if not hasattr(attn, "k_norm"):
                raise RuntimeError(
                    f"layer {i} has no k_norm; this capture assumes Qwen3-style QK-norm"
                )
            self._handles.append(attn.k_norm.register_forward_hook(self._k_hook(i)))
            self._handles.append(attn.v_proj.register_forward_hook(self._v_hook(i)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    def clear(self) -> None:
        self.keys.clear()
        self.values.clear()

    def stacked(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (L, b, kv_heads, seq, head_dim) tensors for keys and values."""
        k = torch.stack([self.keys[i] for i in self.layers])
        v = torch.stack([self.values[i] for i in self.layers])
        return k, v


@torch.no_grad()
def capture_batch(model, input_ids: torch.Tensor, attention_mask=None, store_dtype=torch.float16):
    """One forward pass, returning stacked pre-RoPE (keys, values)."""
    with KVCapture(model, store_dtype=store_dtype) as cap:
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,          # we want the hooks' output, not the cache
        )
        return cap.stacked()
