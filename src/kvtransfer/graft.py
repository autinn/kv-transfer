"""Graft a mapped KV cache into a target model and run decoding from it.

The mapper produces *content-space* keys. The target model's cache expects post-RoPE
keys, so grafting is: map in content space, then rotate with the target's RoPE at the
positions the tokens actually occupy.

    K_hat_t = (K_s R_s^-1(t) W_K + b_K) R_t(t)

Values need no rotation.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache

from .rope import apply_rope, rope_tables_for


def build_cache(
    keys_content: list[torch.Tensor],
    values: list[torch.Tensor],
    head_dim: int,
    rope_theta: float,
    offset: int = 0,
    dtype: torch.dtype | None = None,
    device=None,
    config=None,
) -> DynamicCache:
    """Build a DynamicCache from content-space keys and values.

    keys_content[l], values[l]: (B, H, S, D). Keys are rotated with the target model's
    RoPE here; values are inserted as-is.
    """
    if len(keys_content) != len(values):
        raise ValueError("keys and values must cover the same layers")
    S = keys_content[0].shape[2]
    cos, sin = rope_tables_for(
        S, head_dim, rope_theta, device=keys_content[0].device,
        dtype=torch.float32, offset=offset,
    )
    cache = DynamicCache(config=config) if config is not None else (DynamicCache(config=None) if _accepts_config() else DynamicCache())
    for l, (k, v) in enumerate(zip(keys_content, values)):
        k_rot = apply_rope(k.to(torch.float32), cos, sin)
        if dtype is not None:
            k_rot, v = k_rot.to(dtype), v.to(dtype)
        if device is not None:
            k_rot, v = k_rot.to(device), v.to(device)
        cache.update(k_rot, v, l)
    return cache


def _accepts_config() -> bool:
    import inspect
    try:
        return "config" in inspect.signature(DynamicCache.__init__).parameters
    except (TypeError, ValueError):
        return False


def cache_to_content(cache, head_dim: int, rope_theta: float, offset: int = 0):
    """Inverse of `build_cache`'s rotation: recover content-space keys from a cache."""
    from .rope import strip_rope

    keys, values = [], []
    layers = cache.layers if hasattr(cache, "layers") else None
    n = len(layers) if layers is not None else len(cache)
    for l in range(n):
        k = layers[l].keys if layers is not None else cache[l][0]
        v = layers[l].values if layers is not None else cache[l][1]
        S = k.shape[2]
        cos, sin = rope_tables_for(
            S, head_dim, rope_theta, device=k.device, dtype=torch.float32, offset=offset
        )
        keys.append(strip_rope(k.to(torch.float32), cos, sin))
        values.append(v.to(torch.float32))
    return keys, values


@torch.no_grad()
def logits_from_cache(model, cache, next_token_ids: torch.Tensor, prefix_len: int):
    """Run `next_token_ids` against a pre-populated cache.

    The canonical HF cache contract is to pass only the continuation tokens together
    with the prefilled `past_key_values`; the model itself infers the correct positions
    and causal attention from the cache. Overriding `position_ids` / `attention_mask`
    here can silently break the prefix ordering, which is exactly the kind of bug the
    oracle-graft sanity check is meant to catch.
    """
    out = model(
        input_ids=next_token_ids,
        past_key_values=cache,
        use_cache=True,
    )
    return out.logits
