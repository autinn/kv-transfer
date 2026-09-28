"""Benchmark scoring from a transferred prefix cache: the paper's retention metric.

The paper reports accuracy retention (transfer accuracy / target standalone accuracy)
on ARC-Challenge, HellaSwag, WinoGrande, MMLU and GSM8K. Every task reduces to one of
two operations on a (context, continuation) split:

  * log-likelihood of a continuation given the context (multiple choice)
  * greedy generation after the context (GSM8K)

The context is where transfer happens. It is prefilled by the SOURCE model, mapped, and
grafted into the TARGET, which then reads the continuation. One detail makes this
exact: a cache holds keys and values but no logits, so the prediction for the first
continuation token has to come from a forward pass over the last context token. The
transferred prefix is therefore `context[:-1]`, and the target runs
`[context[-1]] + continuation` on top of it. Standalone scoring runs the whole
`context + continuation` through the target, and must match the oracle path
(target's own cache for `context[:-1]` pushed through the same graft) to numerical
precision -- `tests/test_harness.py` checks that.

Anything that yields per-context content-space KV can drive this: the source store plus
a mapper, the target itself (oracle), or nothing (standalone).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator

import torch

from ..capture import KVCapture
from ..graft import build_cache
from ..runtime import input_device, kv_dims, rope_theta


# --------------------------------------------------------------------------- encoding


def split_context(tok, context: str, continuation: str) -> tuple[list[int], list[int]]:
    """Tokenize a (context, continuation) pair the way lm-evaluation-harness does.

    Trailing spaces move from the context into the continuation, then the continuation
    tokens are whatever the joint encoding adds after the context's tokens. This keeps
    the tokens scored identical to what the model would see in running text.
    """
    n_spaces = len(context) - len(context.rstrip(" "))
    if n_spaces:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole = tok(context + continuation, add_special_tokens=False).input_ids
    ctx = tok(context, add_special_tokens=False).input_ids
    cont = whole[len(ctx):]
    if len(ctx) < 2:
        raise ValueError(f"context too short to transfer (needs >= 2 tokens): {context!r}")
    if not cont:
        raise ValueError(f"empty continuation for {continuation!r}")
    return ctx, cont


@dataclass
class EncodedTask:
    """A task reduced to token ids, with contexts deduplicated.

    `contexts` is the list of unique context token lists, in the order their prefixes
    are stored. For multiple choice, `requests[i]` lists (context index, continuation
    ids, continuation text) per choice of doc i. For generation, `requests[i]` is a
    single context index.
    """

    name: str
    kind: str                      # "mc" | "gen"
    contexts: list[list[int]]
    requests: list
    labels: list                   # int for mc, gold answer string for gen
    metric: str
    gen: dict = field(default_factory=dict)

    @property
    def prefixes(self) -> list[torch.Tensor]:
        """What gets transferred: every context minus its last token."""
        return [torch.tensor(c[:-1]) for c in self.contexts]


def encode_task(task, tok) -> EncodedTask:
    index: dict[tuple, int] = {}
    contexts: list[list[int]] = []

    def ctx_id(ids: list[int]) -> int:
        key = tuple(ids)
        if key not in index:
            index[key] = len(contexts)
            contexts.append(ids)
        return index[key]

    requests, labels = [], []
    for doc in task.docs:
        if task.kind == "mc":
            reqs = []
            for c_text, k_text in zip(doc.contexts, doc.conts):
                ctx, cont = split_context(tok, c_text, k_text)
                reqs.append((ctx_id(ctx), cont, k_text))
            requests.append(reqs)
            labels.append(doc.label)
        else:
            ctx = tok(doc.prompt, add_special_tokens=False).input_ids
            if len(ctx) < 2:
                raise ValueError("generation prompt too short to transfer")
            requests.append(ctx_id(ctx))
            labels.append(doc.answer)
    return EncodedTask(task.name, task.kind, contexts, requests, labels, task.metric, task.gen)


# ------------------------------------------------------------------ prefix providers


def standalone_contents(n: int) -> Iterator[None]:
    """No transfer: the target reads the whole context itself."""
    for _ in range(n):
        yield None


@torch.no_grad()
def oracle_contents(model, contexts: list[list[int]], batch: int = 16) -> Iterator[tuple[list, list]]:
    """The target's own content-space KV for each prefix, through the graft path.

    Prefixes are prefilled `batch` at a time, right-padded (causal attention keeps each
    prefix's KV independent of the padding after it), and yielded one at a time.
    """
    dev = input_device(model)
    for j in range(0, len(contexts), batch):
        pre = [ctx[:-1] for ctx in contexts[j : j + batch]]
        ids = torch.zeros(len(pre), max(len(p) for p in pre), dtype=torch.long)
        for r, p in enumerate(pre):
            ids[r, : len(p)] = torch.tensor(p)
        with KVCapture(model, store_dtype=torch.float32, device=dev) as cap:
            model(input_ids=ids.to(dev), use_cache=False)
        for r, p in enumerate(pre):
            n = len(p)
            yield ([cap.keys[l][r : r + 1, :, :n] for l in cap.layers],
                   [cap.values[l][r : r + 1, :, :n] for l in cap.layers])


def mapped_contents(mapper, store, contexts: list[list[int]]) -> Iterator[tuple[list, list]]:
    """Source prefixes from a store, mapped into the target's content space."""
    from ..kvstore import as_layer_list

    if len(store) != len(contexts):
        raise ValueError(f"store holds {len(store)} prefixes, task has {len(contexts)}")
    for ctx, (k, v) in zip(contexts, store.iter_items()):
        if k.shape[2] != len(ctx) - 1:
            raise ValueError(f"stored prefix has {k.shape[2]} tokens, expected {len(ctx) - 1}")
        yield mapper.map_content(as_layer_list(k), as_layer_list(v))


def _cache_for(model, content, batch: int):
    keys, values = content
    _, _, D = kv_dims(model.config)
    rep = lambda xs: [x.expand(batch, *x.shape[1:]) for x in xs]
    return build_cache(
        rep(keys), rep(values), D, rope_theta(model.config),
        dtype=model.dtype, device=input_device(model), config=model.config,
    )


# -------------------------------------------------------------------------- scoring


@torch.no_grad()
def score_continuations(model, ctx: list[int], conts: list[list[int]], content=None,
                        pad_id: int = 0) -> list[float]:
    """Summed log-probability of each continuation after `ctx`, in one batched forward."""
    return score_batch(model, [(ctx, conts, content)], pad_id)[0]


@torch.no_grad()
def score_batch(model, items: list[tuple], pad_id: int = 0) -> list[list[float]]:
    """`score_continuations` for several (ctx, conts, content) items in one forward.

    Every continuation of every item is one row. Rows are right-padded; with causal
    attention, padding after a row's last real token cannot affect its scored positions.

    With transferred content, each item's prefix KV (length P_i) is right-padded to the
    longest prefix P and the rows run on top of it. A padding mask hides the prefix
    padding, and position ids P_i + j give each row's tokens their true positions, so
    RoPE sees exactly the geometry of an unpadded run: prefix keys are rotated at 0..P_i-1
    and the row's own tokens at P_i.. regardless of which cache slot they occupy.
    """
    dev = input_device(model)
    cached = items[0][2] is not None
    if any((it[2] is not None) != cached for it in items):
        raise ValueError("cannot mix standalone and transferred items in one batch")
    rows, firsts, conts_all = [], [], []
    for ctx, conts, _ in items:
        for c in conts:
            rows.append(ctx + c if not cached else [ctx[-1]] + c)
            firsts.append(len(ctx) - 1 if not cached else 0)   # position predicting c[0]
            conts_all.append(c)
    N, T = len(rows), max(len(r) for r in rows)
    ids = torch.full((N, T), pad_id, dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(r)
    kw = {"use_cache": False}
    if cached:
        _, _, D = kv_dims(model.config)
        plens = [it[2][0][0].shape[2] for it in items]
        P = max(plens)
        reps = [len(it[1]) for it in items]

        def stack(xs_per_item):                   # per item (1, H, P_i, D) -> (N, H, P, D)
            out = []
            for x, n, pl in zip(xs_per_item, reps, plens):
                x = x.to(dev)
                if pl < P:
                    x = torch.nn.functional.pad(x, (0, 0, 0, P - pl))
                out.append(x.expand(n, *x.shape[1:]))
            return torch.cat(out)

        n_layers = len(items[0][2][0])
        keys = [stack([it[2][0][l] for it in items]) for l in range(n_layers)]
        values = [stack([it[2][1][l] for it in items]) for l in range(n_layers)]
        cache = build_cache(keys, values, D, rope_theta(model.config),
                            dtype=model.dtype, device=dev, config=model.config)
        kw = {"use_cache": True, "past_key_values": cache}
        if len(set(plens)) > 1:
            row_plen = torch.tensor([pl for pl, n in zip(plens, reps) for _ in range(n)])
            mask = torch.zeros(N, P + T, dtype=torch.long)
            mask[:, P:] = 1
            mask[:, :P] = (torch.arange(P)[None] < row_plen[:, None]).long()
            kw["attention_mask"] = mask.to(dev)
            kw["position_ids"] = (row_plen[:, None] + torch.arange(T)[None]).to(dev)
    hidden = model.model(input_ids=ids.to(dev), **kw).last_hidden_state   # (N, T, H)
    ri = torch.tensor([i for i, c in enumerate(conts_all) for _ in c], device=dev)
    pi = torch.tensor([f + j for f, c in zip(firsts, conts_all) for j in range(len(c))], device=dev)
    tg = torch.tensor([t for c in conts_all for t in c], device=dev)
    sel = hidden[ri, pi]
    tok_lp = torch.cat([                              # chunked: (tokens, vocab) fp32 is big
        torch.log_softmax(model.lm_head(sel[a : a + 1024]).float(), dim=-1)
        .gather(-1, tg[a : a + 1024, None])[:, 0]
        for a in range(0, len(sel), 1024)
    ])
    per_row = torch.zeros(N, dtype=torch.float64, device=dev).index_add_(0, ri, tok_lp.double())
    per_row = per_row.tolist()
    out, i = [], 0
    for _, conts, _ in items:
        out.append(per_row[i : i + len(conts)])
        i += len(conts)
    return out


@torch.no_grad()
def greedy_generate(model, tok, ctx: list[int], content=None, max_new_tokens: int = 256,
                    stop: Iterable[str] = ()) -> str:
    """Greedy decoding after `ctx`, stopping at EOS or any stop string."""
    dev = input_device(model)
    stop = list(stop)
    eos = {t for t in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                       tok.convert_tokens_to_ids("<|endoftext|>")) if isinstance(t, int) and t >= 0}
    if content is None:
        out = model(input_ids=torch.tensor([ctx], device=dev), use_cache=True, logits_to_keep=1)
    else:
        out = model(input_ids=torch.tensor([[ctx[-1]]], device=dev),
                    past_key_values=_cache_for(model, content, 1), use_cache=True)
    cache = out.past_key_values
    gen: list[int] = []
    text = ""
    for _ in range(max_new_tokens):
        nxt = int(out.logits[0, -1].argmax())
        if nxt in eos:
            break
        gen.append(nxt)
        text = tok.decode(gen, skip_special_tokens=True)
        if any(s in text for s in stop):
            break
        out = model(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=cache, use_cache=True)
    for s in stop:
        text = text.split(s)[0]
    return text


# -------------------------------------------------------------------------- metrics


def _norm_number(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def extract_answer(text: str, mode: str = "flexible") -> str | None:
    """GSM8K answer extraction, following lm-evaluation-harness's two filters."""
    if mode == "strict":
        m = re.search(r"#### (\-?[0-9\.\,]+)", text)
        return _norm_number(m.group(1)) if m else None
    found = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", text)
    for a, b in reversed(found):
        cand = (a or b).strip()
        if re.search(r"[0-9]", cand):
            return _norm_number(cand)
    return None


def gold_answer(answer: str) -> str:
    return _norm_number(answer.split("####")[-1])


def score_contexts(model, tok, task: EncodedTask, ctx_ids: list[int], contents: Iterator,
                   log_every: int = 0, batch_tokens: int = 8192) -> dict:
    """Raw scores for a subset of a task's contexts, one prefix content per context.

    Returns {context index: [log-likelihood per request on that context]} for multiple
    choice, or {context index: generated text} for generation. Results for disjoint
    subsets merge with `dict.update`, which is what lets a large task be evaluated in
    disk-sized rounds and resumed.

    Multiple-choice contexts are scored several per forward, up to about `batch_tokens`
    padded row tokens (0 = one context per forward).
    """
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    by_ctx = task_requests_by_context(task) if task.kind == "mc" else None
    out: dict[int, object] = {}
    n = 0
    buf, buf_rows, buf_T = [], 0, 0

    def flush():
        nonlocal buf, buf_rows, buf_T
        if buf:
            for ci, s in zip([b[0] for b in buf], score_batch(model, [b[1] for b in buf], pad_id)):
                out[ci] = s
        buf, buf_rows, buf_T = [], 0, 0

    for ci, content in zip(ctx_ids, contents):
        n += 1
        if task.kind == "mc":
            ctx = task.contexts[ci]
            conts = [task.requests[d][c][1] for d, c in by_ctx.get(ci, [])]
            # A transferred row is short, but its prefix cache costs as much as the
            # standalone row's context, so both are budgeted at context + continuation.
            T = max(len(c) for c in conts) + len(ctx)
            if buf and (buf_rows + len(conts)) * max(buf_T, T) > batch_tokens:
                flush()
            buf.append((ci, (ctx, conts, content)))
            buf_rows += len(conts)
            buf_T = max(buf_T, T)
        else:
            out[ci] = greedy_generate(model, tok, task.contexts[ci], content,
                                      task.gen.get("max_new_tokens", 256), task.gen.get("stop", ()))
        if log_every and n % log_every == 0:
            print(f"      {n}/{len(ctx_ids)} contexts", flush=True)
    flush()
    if n != len(ctx_ids):
        raise ValueError(f"prefix provider yielded {n} contents for {len(ctx_ids)} contexts")
    return out


def task_requests_by_context(task: EncodedTask) -> dict[int, list[tuple[int, int]]]:
    """context index -> [(doc, choice)], in a fixed order shared with `finalize`."""
    by_ctx: dict[int, list[tuple[int, int]]] = {}
    for d, reqs in enumerate(task.requests):
        for c, (ci, _, _) in enumerate(reqs):
            by_ctx.setdefault(ci, []).append((d, c))
    return by_ctx


def finalize(task: EncodedTask, raw: dict) -> dict:
    """Metrics from the merged raw scores of every context."""
    missing = [ci for ci in range(len(task.contexts)) if ci not in raw]
    if missing:
        raise ValueError(f"{task.name}: {len(missing)} contexts unscored, e.g. {missing[0]}")
    n = len(task.requests)
    if task.kind == "mc":
        ll = [[0.0] * len(r) for r in task.requests]
        for ci, pairs in task_requests_by_context(task).items():
            for (d, c), s in zip(pairs, raw[ci]):
                ll[d][c] = s
        acc = acc_norm = 0
        for d, reqs in enumerate(task.requests):
            lens = [len(r[2]) for r in reqs]
            acc += int(max(range(len(reqs)), key=lambda c: ll[d][c]) == task.labels[d])
            acc_norm += int(max(range(len(reqs)), key=lambda c: ll[d][c] / lens[c]) == task.labels[d])
        return {"acc": acc / n, "acc_norm": acc_norm / n, "n": n}
    strict = flex = 0
    for d, ci in enumerate(task.requests):
        gold = gold_answer(task.labels[d])
        strict += int(extract_answer(raw[ci], "strict") == gold)
        flex += int(extract_answer(raw[ci], "flexible") == gold)
    return {"exact_match_strict": strict / n, "exact_match_flexible": flex / n, "n": n}


def run_task(model, tok, task: EncodedTask, contents: Iterator, log_every: int = 0) -> dict:
    """Score every context of a task in order and return its metrics."""
    ids = list(range(len(task.contexts)))
    return finalize(task, score_contexts(model, tok, task, ids, contents, log_every))
