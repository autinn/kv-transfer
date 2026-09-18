"""Check our RoPE against the real HF Qwen3 implementation, not against itself.

A self-consistent strip/apply pair can still be the *wrong rotation* (e.g. GPT-J
interleaved instead of GPT-NeoX half-split) and would round-trip perfectly while
corrupting every key that crosses between our code and the model's cache.
"""

import torch
from transformers.models.qwen3 import modeling_qwen3 as hf

from kvtransfer.rope import apply_rope, rope_tables_for, rotate_half, strip_rope

HEAD_DIM, THETA, SEQ, HEADS = 128, 1_000_000.0, 64, 8


def test_rotate_half_matches_hf():
    x = torch.randn(2, HEADS, SEQ, HEAD_DIM, dtype=torch.float64)
    assert torch.equal(rotate_half(x), hf.rotate_half(x))


def test_apply_rope_matches_hf():
    """Our apply_rope must equal HF's apply_rotary_pos_emb given the same tables."""
    torch.manual_seed(0)
    q = torch.randn(1, HEADS, SEQ, HEAD_DIM, dtype=torch.float32)
    k = torch.randn(1, HEADS, SEQ, HEAD_DIM, dtype=torch.float32)

    cos, sin = rope_tables_for(SEQ, HEAD_DIM, THETA, dtype=torch.float32)
    # HF expects (batch, seq, head_dim) and unsqueezes dim 1 internally.
    hf_q, hf_k = hf.apply_rotary_pos_emb(q, k, cos[:, 0], sin[:, 0])

    assert torch.allclose(apply_rope(q, cos, sin), hf_q, atol=1e-6)
    assert torch.allclose(apply_rope(k, cos, sin), hf_k, atol=1e-6)


def test_strip_inverts_apply():
    torch.manual_seed(1)
    x = torch.randn(1, HEADS, SEQ, HEAD_DIM, dtype=torch.float64)
    cos, sin = rope_tables_for(SEQ, HEAD_DIM, THETA, dtype=torch.float64)
    assert torch.allclose(strip_rope(apply_rope(x, cos, sin), cos, sin), x, atol=1e-10)


def test_strip_inverts_at_long_offsets():
    """Position independence must hold well past the 1024-token calibration length."""
    x = torch.randn(1, HEADS, SEQ, HEAD_DIM, dtype=torch.float64)
    for offset in (0, 1024, 8192, 32768):
        cos, sin = rope_tables_for(SEQ, HEAD_DIM, THETA, dtype=torch.float64, offset=offset)
        back = strip_rope(apply_rope(x, cos, sin), cos, sin)
        assert torch.allclose(back, x, atol=1e-9), f"failed at offset {offset}"


def test_content_space_is_position_invariant():
    """The property the whole method rests on.

    Two copies of the same content at different positions must collapse to the
    *same* vector once RoPE is stripped. If this fails, a mapper fitted at
    positions 0..1023 cannot generalize to position 4096.
    """
    content = torch.randn(1, HEADS, SEQ, HEAD_DIM, dtype=torch.float64)
    outs = []
    for offset in (0, 777, 4096):
        cos, sin = rope_tables_for(SEQ, HEAD_DIM, THETA, dtype=torch.float64, offset=offset)
        outs.append(strip_rope(apply_rope(content, cos, sin), cos, sin))
    assert torch.allclose(outs[0], outs[1], atol=1e-9)
    assert torch.allclose(outs[0], outs[2], atol=1e-9)


def test_interleaved_convention_would_differ():
    """Guard against silently swapping in the GPT-J convention.

    Documents that the two conventions are genuinely different rotations, so this
    test suite would catch the substitution.
    """
    x = torch.randn(1, 1, SEQ, HEAD_DIM, dtype=torch.float64)
    cos, sin = rope_tables_for(SEQ, HEAD_DIM, THETA, dtype=torch.float64, offset=1)

    def rotate_half_interleaved(t):
        t = t.reshape(*t.shape[:-1], -1, 2)
        return torch.stack((-t[..., 1], t[..., 0]), dim=-1).reshape(*t.shape[:-2], -1)

    neox = x * cos + rotate_half(x) * sin
    gptj = x * cos + rotate_half_interleaved(x) * sin
    assert not torch.allclose(neox, gptj, atol=1e-6)
