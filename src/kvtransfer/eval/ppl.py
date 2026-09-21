"""Prefix-conditioned perplexity: the primary transfer metric.

Split each sequence into a prefix and a continuation. The prefix is prefilled by the
SOURCE model, mapped into the target's format, and grafted in; the continuation is then
scored by the TARGET model. Perplexity on the continuation measures how much usable
information survived the mapping.

This is the cheapest sensitive metric available. Multiple-choice benchmarks can score
above chance on a badly damaged cache, but perplexity degrades smoothly and immediately,
so it is the right first read on whether a mapper works.
"""

from __future__ import annotations

import torch

from ..capture import KVCapture
from ..graft import build_cache, logits_from_cache


@torch.no_grad()
def _nll(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, int]:
    """Sum of token NLLs and the token count, computed in float32."""
    lp = torch.log_softmax(logits.float(), dim=-1)
    tok_lp = lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(-tok_lp.sum()), int(targets.numel())


@torch.no_grad()
def standalone_ppl(model, ids: torch.Tensor, prefix_len: int, device="cpu") -> float:
    """Target model prefills its own prefix. This is the ceiling any mapper is measured against."""
    total, n = 0.0, 0
    for seq in ids:
        seq = seq.unsqueeze(0).to(device)
        # Cache replay receives the first continuation token as input, so its logits
        # begin by predicting the second continuation token. Score that same range
        # here so standalone, oracle, and transfer PPL use identical targets.
        logits = model(input_ids=seq).logits[:, prefix_len : -1]
        targets = seq[:, prefix_len + 1 :]
        if targets.numel() == 0:
            continue
        s, c = _nll(logits, targets)
        total += s
        n += c
    return float(torch.exp(torch.tensor(total / n)))


@torch.no_grad()
def source_prefill_content(src_model, seq: torch.Tensor, prefix_len: int, n_layers: int):
    """Run the source model over the prefix and return content-space (pre-RoPE) KV."""
    with KVCapture(src_model, store_dtype=torch.float32) as cap:
        src_model(input_ids=seq[:, :prefix_len], use_cache=False)
        return (
            [cap.keys[l] for l in range(n_layers)],
            [cap.values[l] for l in range(n_layers)],
        )


@torch.no_grad()
def transfer_ppl(
    tgt_model,
    mapper,
    src_caches,
    ids: torch.Tensor,
    prefix_len: int,
    device="cpu",
) -> float:
    """Perplexity on the continuation, given a mapped-and-grafted prefix cache.

    `src_caches` is a list of (keys, values) content-space captures from the source
    model, one per sequence -- precomputed so the source model need not be resident
    while the target model runs.
    """
    total, n = 0.0, 0
    for seq, (sk, sv) in zip(ids, src_caches):
        seq = seq.unsqueeze(0)
        cache = mapper.to_cache(sk, sv, offset=0, dtype=tgt_model.dtype, device=device, config=tgt_model.config)
        rest = seq[:, prefix_len:].to(device)
        if rest.numel() == 0:
            continue
        logits = logits_from_cache(tgt_model, cache, rest, prefix_len=prefix_len)
        # The continuation is scored on the next-token predictions conditioned on the
        # prefix and each preceding continuation token; the model emits one logit for
        # each input token, but the target for token j is the next token, i.e. rest[:,1:].
        s, c = _nll(logits[:, :-1], rest[:, 1:])
        total += s
        n += c
        del cache
    return float(torch.exp(torch.tensor(total / n)))


@torch.no_grad()
def oracle_ppl(tgt_model, ids: torch.Tensor, prefix_len: int, n_layers: int, device="cpu") -> float:
    """Target prefills its own prefix, but routed through the graft path.

    Separates "the mapping lost information" from "the grafting machinery is broken":
    this must match `standalone_ppl` almost exactly, and if it does not, the harness is
    at fault rather than the mapper.
    """
    cfg = tgt_model.config
    theta = float(cfg.rope_parameters["rope_theta"])
    total, n = 0.0, 0
    for seq in ids:
        seq = seq.unsqueeze(0)
        with KVCapture(tgt_model, store_dtype=torch.float32) as cap:
            tgt_model(input_ids=seq[:, :prefix_len].to(device), use_cache=False)
            k = [cap.keys[l] for l in range(n_layers)]
            v = [cap.values[l] for l in range(n_layers)]
        cache = build_cache(k, v, cfg.head_dim, theta, dtype=tgt_model.dtype, device=device, config=tgt_model.config)
        rest = seq[:, prefix_len:].to(device)
        if rest.numel() == 0:
            continue
        logits = logits_from_cache(tgt_model, cache, rest, prefix_len=prefix_len)
        s, c = _nll(logits[:, :-1], rest[:, 1:])
        total += s
        n += c
        del cache
    return float(torch.exp(torch.tensor(total / n)))
