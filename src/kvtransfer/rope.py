"""RoPE strip / re-apply in the non-interleaved (GPT-NeoX) convention Qwen3 uses.

The paper maps KV caches in *content space*: source keys are un-rotated, the linear
map is fitted there, and the target model's rotation is applied to the result.

    K_hat_t = (K_s R_s^-1(t) W_K + b_K) R_t(t)

Values carry no positional encoding and map directly.

Convention matters. Qwen3's `apply_rotary_pos_emb` uses `rotate_half`, which splits
the head dimension in half (dims i and i + D/2 are paired). The GPT-J style that pairs
adjacent dims (2i, 2i+1) is a *different* rotation. Using the wrong one still runs and
still produces plausible-looking numbers, so `tests/test_rope.py` checks ours against
the real HF implementation rather than against itself.
"""

from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Identical to `transformers.models.qwen3.modeling_qwen3.rotate_half`."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def inv_freq(head_dim: int, theta: float, device=None) -> torch.Tensor:
    """Inverse frequencies, matching Qwen3RotaryEmbedding.compute_default_rope_parameters."""
    return 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )


def cos_sin(
    positions: torch.Tensor,
    head_dim: int,
    theta: float,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables for `positions`, shape (..., head_dim).

    Computed in float32 regardless of the requested output dtype: HF forces float32
    here too, because bf16 trig at position ~10^4 loses unacceptable precision.
    """
    freqs = positions.to(torch.float32).unsqueeze(-1) * inv_freq(
        head_dim, theta, device=positions.device
    )
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate `x` (..., seq, head_dim) forward by the given cos/sin."""
    return x * cos + rotate_half(x) * sin


def strip_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Inverse rotation.

    R(t)^-1 == R(-t), and since cos is even and sin is odd this is just `sin -> -sin`.
    """
    return x * cos + rotate_half(x) * (-sin)


def rope_tables_for(
    seq_len: int,
    head_dim: int,
    theta: float,
    device=None,
    dtype: torch.dtype = torch.float32,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin for positions [offset, offset + seq_len), broadcastable over heads.

    Returns shape (1, 1, seq_len, head_dim) to line up with (batch, heads, seq, dim).
    """
    pos = torch.arange(offset, offset + seq_len, device=device)
    cos, sin = cos_sin(pos, head_dim, theta, dtype=dtype)
    return cos[None, None], sin[None, None]
