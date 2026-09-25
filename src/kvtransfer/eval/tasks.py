"""The paper's five benchmarks, formatted as lm-evaluation-harness formats them.

The paper does not publish its prompts. These follow lm-evaluation-harness's default
task definitions (the de facto standard), so that absolute accuracies are comparable
with the usual published numbers. Retention is a ratio of two runs under the same
prompts, so it is much less sensitive to formatting than absolute accuracy.

One deliberate difference: few-shot examples are one fixed sample per task (seeded)
rather than resampled per document. That keeps every document's context identical in
its few-shot part, which is simpler to reason about and does not change the metric.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field


@dataclass
class MCDoc:
    contexts: list[str]            # one per choice (identical except for WinoGrande)
    conts: list[str]
    label: int


@dataclass
class GenDoc:
    prompt: str
    answer: str


@dataclass
class Task:
    name: str
    kind: str                      # "mc" | "gen"
    docs: list
    metric: str                    # the number retention is computed on
    gen: dict = field(default_factory=dict)


def _load(path, name, split):
    from datasets import load_dataset

    return load_dataset(path, name, split=split)


def _fewshot(pool, n: int, seed: int) -> list:
    if n <= 0:
        return []
    idx = random.Random(seed).sample(range(len(pool)), n)
    return [pool[i] for i in idx]


def _join(shots: list[str], query: str, description: str = "") -> str:
    return description + "".join(s + "\n\n" for s in shots) + query


# ------------------------------------------------------------------------ HellaSwag


def _hs_pre(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    return text.replace("  ", " ")


def _hs_query(d) -> tuple[str, list[str]]:
    ctx = d["ctx_a"] + " " + d["ctx_b"].capitalize()
    return _hs_pre(d["activity_label"] + ": " + ctx), [_hs_pre(e) for e in d["endings"]]


def hellaswag(num_fewshot=0, limit=None, seed=1234) -> Task:
    ds = _load("Rowan/hellaswag", None, "validation")
    shots = []
    for d in _fewshot(_load("Rowan/hellaswag", None, "train"), num_fewshot, seed):
        q, ends = _hs_query(d)
        shots.append(q + " " + ends[int(d["label"])])
    docs = []
    for d in ds.select(range(min(limit or len(ds), len(ds)))):
        q, ends = _hs_query(d)
        ctx = _join(shots, q)
        docs.append(MCDoc([ctx] * len(ends), [" " + e for e in ends], int(d["label"])))
    return Task("hellaswag", "mc", docs, "acc_norm")


# ------------------------------------------------------------------- ARC-Challenge


def _arc_query(d) -> str:
    return f"Question: {d['question']}\nAnswer:"


def arc_challenge(num_fewshot=0, limit=None, seed=1234) -> Task:
    ds = _load("allenai/ai2_arc", "ARC-Challenge", "test")
    shots = []
    for d in _fewshot(_load("allenai/ai2_arc", "ARC-Challenge", "train"), num_fewshot, seed):
        i = d["choices"]["label"].index(d["answerKey"])
        shots.append(_arc_query(d) + " " + d["choices"]["text"][i])
    docs = []
    for d in ds.select(range(min(limit or len(ds), len(ds)))):
        ctx = _join(shots, _arc_query(d))
        texts = d["choices"]["text"]
        docs.append(MCDoc([ctx] * len(texts), [" " + t for t in texts],
                          d["choices"]["label"].index(d["answerKey"])))
    return Task("arc_challenge", "mc", docs, "acc_norm")


# ---------------------------------------------------------------------- WinoGrande


def _wg_split(d) -> tuple[list[str], str]:
    """Partial evaluation: the option goes in the context, the shared rest is scored."""
    i = d["sentence"].index("_")
    ctxs = [d["sentence"][:i] + d["option1"], d["sentence"][:i] + d["option2"]]
    return ctxs, " " + d["sentence"][i + 1 :].strip()


def winogrande(num_fewshot=0, limit=None, seed=1234) -> Task:
    ds = _load("allenai/winogrande", "winogrande_xl", "validation")
    shots = []
    for d in _fewshot(_load("allenai/winogrande", "winogrande_xl", "train"), num_fewshot, seed):
        ctxs, rest = _wg_split(d)
        shots.append(ctxs[int(d["answer"]) - 1] + rest)
    docs = []
    for d in ds.select(range(min(limit or len(ds), len(ds)))):
        ctxs, rest = _wg_split(d)
        docs.append(MCDoc([_join(shots, c) for c in ctxs], [rest, rest], int(d["answer"]) - 1))
    return Task("winogrande", "mc", docs, "acc")


# ---------------------------------------------------------------------------- MMLU


def _mmlu_query(d) -> str:
    opts = "".join(f"\n{l}. {c}" for l, c in zip("ABCD", d["choices"]))
    return f"{d['question'].strip()}{opts}\nAnswer:"


def mmlu(num_fewshot=0, limit=None, seed=1234) -> Task:
    ds = _load("cais/mmlu", "all", "test")
    dev: dict[str, list] = {}
    if num_fewshot:
        for d in _load("cais/mmlu", "all", "dev"):
            dev.setdefault(d["subject"], []).append(d)
    if limit and limit < len(ds):
        # MMLU is sorted by subject; a head slice would be 2-3 subjects. Sample evenly.
        idx = sorted(random.Random(seed).sample(range(len(ds)), limit))
        ds = ds.select(idx)
    docs = []
    for d in ds:
        subj = d["subject"]
        desc = f"The following are multiple choice questions (with answers) about {subj.replace('_', ' ')}.\n\n"
        shots = [_mmlu_query(s) + " " + "ABCD"[s["answer"]] for s in dev.get(subj, [])[:num_fewshot]]
        ctx = _join(shots, _mmlu_query(d), desc)
        docs.append(MCDoc([ctx] * 4, [" A", " B", " C", " D"], int(d["answer"])))
    return Task("mmlu", "mc", docs, "acc")


# --------------------------------------------------------------------------- GSM8K


def gsm8k(num_fewshot=5, limit=None, seed=1234, max_new_tokens=256) -> Task:
    ds = _load("openai/gsm8k", "main", "test")
    shots = [f"Question: {d['question']}\nAnswer: {d['answer']}"
             for d in _fewshot(_load("openai/gsm8k", "main", "train"), num_fewshot, seed)]
    docs = [GenDoc(_join(shots, f"Question: {d['question']}\nAnswer:"), d["answer"])
            for d in ds.select(range(min(limit or len(ds), len(ds))))]
    return Task("gsm8k", "gen", docs, "exact_match_flexible",
                gen={"max_new_tokens": max_new_tokens, "stop": ["Question:", "</s>", "<|im_end|>"]})


TASKS = {
    "hellaswag": hellaswag,
    "arc_challenge": arc_challenge,
    "winogrande": winogrande,
    "mmlu": mmlu,
    "gsm8k": gsm8k,
}


def load_task(name: str, **kw) -> Task:
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; choose from {sorted(TASKS)}")
    return TASKS[name](**{k: v for k, v in kw.items() if v is not None})
