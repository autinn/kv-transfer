"""Tiny random-weight Qwen3 models for testing without downloads.

The pair deliberately has DIFFERENT depths (source 2 layers, target 3), the shape of
the 14B (40) -> 32B (64) pair, so every code path that used to assume l -> l layer
correspondence is exercised by the fast suite.
"""

import pytest
import torch

VOCAB = 97


def tiny_qwen3(n_layers: int, seed: int, vocab: int = VOCAB, hidden: int = 32):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=2 * hidden,
        num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=512, rope_theta=1e6, tie_word_embeddings=True,
    )
    m = Qwen3ForCausalLM(cfg).eval()
    return m


@pytest.fixture(scope="session")
def tiny_pair():
    """(source, target) with 2 and 3 layers, 2 KV heads x 16 dims each."""
    return tiny_qwen3(2, seed=0), tiny_qwen3(3, seed=1)


class CharTokenizer:
    """Minimal tokenizer: one id per character, enough for harness logic tests."""

    eos_token_id = 1
    pad_token_id = 0

    def __init__(self, vocab: int = VOCAB):
        self.vocab = vocab

    def __call__(self, text, add_special_tokens=False):
        class Out:
            pass

        o = Out()
        o.input_ids = [2 + (ord(ch) % (self.vocab - 2)) for ch in text]
        return o

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + (i % 26)) for i in ids)

    def convert_tokens_to_ids(self, token):
        return None
