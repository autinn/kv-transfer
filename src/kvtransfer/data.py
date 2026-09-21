"""Calibration and evaluation corpora.

The paper calibrates on 500 FineWeb-Edu sequences of 1024 tokens. Both Qwen3 models
share a tokenizer, so a sequence is tokenized once and fed to both -- which is not a
convenience but a requirement: the mapper regresses target KV on source KV *token by
token*, so the two models must see identical token streams.
"""

from __future__ import annotations

import torch


def _pack(token_lists, seq_len: int, n_seqs: int) -> torch.Tensor:
    """Concatenate documents and chop into fixed-length sequences.

    Packing rather than padding keeps every position a real token, so no observation
    in the regression is a pad artifact.
    """
    buf, out = [], []
    for toks in token_lists:
        buf.extend(toks)
        while len(buf) >= seq_len and len(out) < n_seqs:
            out.append(buf[:seq_len])
            buf = buf[seq_len:]
        if len(out) >= n_seqs:
            break
    if not out:
        raise RuntimeError("no sequences produced; corpus too small")
    return torch.tensor(out, dtype=torch.long)


def fineweb_edu(
    tokenizer,
    n_seqs: int = 500,
    seq_len: int = 1024,
    seed: int = 0,
    cache_dir: str | None = None,
) -> torch.Tensor:
    """Calibration set: FineWeb-Edu, streamed so nothing large hits disk.

    Deliberately NOT using `.shuffle(buffer_size=...)`: a shuffle buffer must fill
    before the iterator yields anything, which downloads hundreds of megabytes even
    when only a handful of sequences are wanted (measured: 12 minutes and 270 MB for
    a 4-sequence smoke test). Documents in the shard are already unordered with
    respect to anything we care about, so we take a contiguous run and vary `seed` by
    skipping a deterministic offset instead.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
        cache_dir=cache_dir,
    )
    if seed:
        ds = ds.skip(seed * 1000)

    needed = n_seqs * seq_len
    toks, total = [], 0
    for rec in ds:
        ids = tokenizer(rec["text"], add_special_tokens=False).input_ids
        if not ids:
            continue
        toks.append(ids)
        total += len(ids)
        if total >= needed + seq_len:
            break
    return _pack(toks, seq_len, n_seqs)


def wikitext2(tokenizer, n_seqs: int = 64, seq_len: int = 1024) -> torch.Tensor:
    """Held-out evaluation set, from a different domain than calibration.

    Using a different corpus for eval is what makes the retention number meaningful:
    a mapper that only works on its calibration domain has not learned the cross-model
    relationship, only the calibration text.
    """
    from datasets import load_dataset

    ds = None
    last_err = None
    for repo, config in [
        ("Salesforce/wikitext", "wikitext-2-raw-v1"),
        ("wikitext", "wikitext-2-raw-v1"),
    ]:
        try:
            ds = load_dataset(repo, config, split="test")
            break
        except Exception as e:  # pragma: no cover - compatibility shim for HF versions
            last_err = e
    if ds is None:
        raise last_err

    texts = ds["text"] if "text" in ds.column_names else ds[0].keys()
    text = "\n\n".join(str(t) for t in texts if str(t).strip())
    if not text:
        raise RuntimeError("WikiText test split is empty")
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return _pack([ids], seq_len, n_seqs)


def batches(ids: torch.Tensor, batch_size: int):
    """Yield (B, S) batches."""
    for i in range(0, ids.shape[0], batch_size):
        yield ids[i : i + batch_size]


def cached_calibration(
    tokenizer, n_seqs: int, seq_len: int, seed: int, cache_path
) -> torch.Tensor:
    """Tokenize once, reuse thereafter.

    The token ids are small (500 x 1024 x int64 = 4 MB) but producing them means
    streaming and tokenizing tens of megabytes of text. Caching keeps repeated runs
    fast and guarantees every stage sees byte-identical calibration data.
    """
    from pathlib import Path

    p = Path(cache_path)
    if p.exists():
        ids = torch.load(p)
        if tuple(ids.shape) == (n_seqs, seq_len):
            return ids
    ids = fineweb_edu(tokenizer, n_seqs=n_seqs, seq_len=seq_len, seed=seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ids, p)
    return ids
