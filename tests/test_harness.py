"""Benchmark harness: transfer scoring must agree with the target reading its own context.

Run on the tiny random 3-layer target, so the checks are about the scoring machinery,
not about any model's accuracy.
"""

import pytest
import torch

from conftest import CharTokenizer
from kvtransfer.eval import tasks as T
from kvtransfer.eval.harness import (
    EncodedTask, encode_task, extract_answer, finalize, gold_answer, greedy_generate,
    mapped_contents, oracle_contents, run_task, score_contexts, score_continuations,
    split_context, standalone_contents,
)
from kvtransfer.kvstore import build_store
from kvtransfer.mapper import ZeroCopyMapper


def _oracle(model, ctx):
    return next(oracle_contents(model, [ctx]))


def test_oracle_scores_match_standalone(tiny_pair):
    _, tgt = tiny_pair
    ctx = list(range(5, 25))
    conts = [[30, 31, 32], [40], [50, 51, 52, 53, 54]]
    a = score_continuations(tgt, ctx, conts)
    b = score_continuations(tgt, ctx, conts, _oracle(tgt, ctx))
    assert a == pytest.approx(b, abs=1e-4)


def test_padding_does_not_change_scores(tiny_pair):
    _, tgt = tiny_pair
    ctx = list(range(5, 25))
    alone = score_continuations(tgt, ctx, [[40]], _oracle(tgt, ctx))
    batched = score_continuations(tgt, ctx, [[40], [50, 51, 52, 53, 54, 55]], _oracle(tgt, ctx))
    assert alone[0] == pytest.approx(batched[0], abs=1e-4)


def test_oracle_generation_matches_standalone(tiny_pair):
    _, tgt = tiny_pair
    tok = CharTokenizer()
    ctx = list(range(5, 30))
    a = greedy_generate(tgt, tok, ctx, None, max_new_tokens=12)
    b = greedy_generate(tgt, tok, ctx, _oracle(tgt, ctx), max_new_tokens=12)
    assert a == b and len(a) > 0


def test_self_transfer_through_store_matches_oracle(tiny_pair, tmp_path):
    """Target -> store -> identity mapper -> target reproduces standalone scores."""
    _, tgt = tiny_pair
    ctxs = [list(range(5, 20)), list(range(7, 30))]
    store = build_store(lambda: tgt, "tiny-t", tgt.config, [torch.tensor(c[:-1]) for c in ctxs],
                        tmp_path / "s", store_dtype="float32", log=lambda *_: None)
    zc = ZeroCopyMapper(3, 16, 2, 1e6, n_src_layers=3)
    for ctx, content in zip(ctxs, mapped_contents(zc, store, ctxs)):
        conts = [[60, 61], [70]]
        assert score_continuations(tgt, ctx, conts, content) == pytest.approx(
            score_continuations(tgt, ctx, conts), abs=1e-4)


def test_mapped_contents_rejects_wrong_store(tiny_pair, tmp_path):
    _, tgt = tiny_pair
    store = build_store(lambda: tgt, "tiny-t", tgt.config, [torch.arange(2, 12)],
                        tmp_path / "s", log=lambda *_: None)
    zc = ZeroCopyMapper(3, 16, 2, 1e6, n_src_layers=3)
    with pytest.raises(ValueError, match="store holds 1"):
        list(mapped_contents(zc, store, [list(range(11)), list(range(11))]))
    with pytest.raises(ValueError, match="expected"):
        list(mapped_contents(zc, store, [list(range(20))]))


def test_split_context_moves_trailing_space():
    tok = CharTokenizer()
    ctx, cont = split_context(tok, "Question: x\nAnswer: ", "yes")
    assert ctx == tok("Question: x\nAnswer:").input_ids
    assert cont == tok(" yes").input_ids
    with pytest.raises(ValueError, match="too short"):
        split_context(tok, "a", " b")


def _mc_task():
    docs = [T.MCDoc(["Q one?"] * 3, [" a", " bb", " ccc"], 2),
            T.MCDoc(["Q two?"] * 3, [" a", " bb", " ccc"], 0)]
    return T.Task("toy", "mc", docs, "acc")


def test_encode_dedups_shared_contexts():
    et = encode_task(_mc_task(), CharTokenizer())
    assert len(et.contexts) == 2
    assert [len(r) for r in et.requests] == [3, 3]
    assert all(len(p) == len(c) - 1 for p, c in zip(et.prefixes, et.contexts))


def test_finalize_acc_and_acc_norm():
    et = encode_task(_mc_task(), CharTokenizer())
    # doc 0: raw ll favours " a", per-char favours " ccc" (label 2)
    # doc 1: " a" wins both ways (label 0)
    raw = {0: [-2.0, -5.0, -3.0], 1: [-1.0, -9.0, -9.0]}
    m = finalize(et, raw)
    assert m["acc"] == 0.5 and m["acc_norm"] == 1.0
    with pytest.raises(ValueError, match="unscored"):
        finalize(et, {0: raw[0]})


def test_scoring_in_rounds_equals_one_pass(tiny_pair):
    _, tgt = tiny_pair
    tok = CharTokenizer()
    et = encode_task(_mc_task(), tok)
    whole = run_task(tgt, tok, et, standalone_contents(2))
    raw = score_contexts(tgt, tok, et, [1], standalone_contents(1))
    raw.update(score_contexts(tgt, tok, et, [0], standalone_contents(1)))
    assert finalize(et, raw) == whole
    with pytest.raises(ValueError, match="yielded 1 contents for 2"):
        score_contexts(tgt, tok, et, [0, 1], standalone_contents(1))


def test_gsm8k_answer_extraction():
    assert gold_answer("She pays 3 * 4 = 12.\n#### 1,234") == "1234"
    assert extract_answer("so the total is $1,234.", "flexible") == "1234"
    assert extract_answer("#### 72", "strict") == "72"
    assert extract_answer("the answer is 72", "strict") is None
    assert extract_answer("3 apples then 5.0 left", "flexible") == "5"
    assert extract_answer("no digits here", "flexible") is None


def test_generation_task_end_to_end(tiny_pair):
    _, tgt = tiny_pair
    tok = CharTokenizer()
    task = T.Task("g", "gen", [T.GenDoc("Question: 1+1?\nAnswer:", "#### 2")], "exact_match_flexible",
                  gen={"max_new_tokens": 4, "stop": ["Question:"]})
    et = encode_task(task, tok)
    m = run_task(tgt, tok, et, oracle_contents(tgt, et.contexts))
    assert set(m) == {"exact_match_strict", "exact_match_flexible", "n"} and m["n"] == 1


# ------------------------------------------------------------ prompt formatting


def test_hellaswag_preprocessing():
    d = {"activity_label": "Roof", "ctx_a": "A man [title] climbs.", "ctx_b": "he", "endings": ["x  y", " z "]}
    q, ends = T._hs_query(d)
    assert q == "Roof: A man. climbs. He"
    assert ends == ["x y", "z"]


def test_winogrande_partial_split():
    d = {"sentence": "The cup fell off the table because _ was wobbly.", "option1": "the cup",
         "option2": "the table", "answer": "2"}
    ctxs, rest = T._wg_split(d)
    assert ctxs == ["The cup fell off the table because the cup",
                    "The cup fell off the table because the table"]
    assert rest == " was wobbly."


def test_mmlu_prompt():
    d = {"question": " What is 2+2? ", "choices": ["1", "2", "3", "4"], "answer": 3}
    assert T._mmlu_query(d) == "What is 2+2?\nA. 1\nB. 2\nC. 3\nD. 4\nAnswer:"


def test_fewshot_is_deterministic():
    pool = list(range(100))
    assert T._fewshot(pool, 5, 1234) == T._fewshot(pool, 5, 1234)
    assert T._fewshot(pool, 0, 1234) == []


def test_unknown_task():
    with pytest.raises(KeyError, match="unknown task"):
        T.load_task("nope")


def _batch_task(ctxs, conts_per_ctx):
    reqs = [[(ci, c, 0) for c in conts] for ci, conts in enumerate(conts_per_ctx)]
    return EncodedTask(name="t", kind="mc", contexts=ctxs, requests=reqs, labels=[0] * len(reqs), metric="acc")


@pytest.mark.parametrize("mode", ["standalone", "oracle", "store"])
def test_batched_scoring_matches_one_context_at_a_time(tiny_pair, tmp_path, mode):
    """Contexts of different lengths scored together (padded prefix caches, padding
    mask, per-row positions) must give the per-context scores."""
    _, tgt = tiny_pair
    g = torch.Generator().manual_seed(0)
    ctxs = [torch.randint(2, 90, (n,), generator=g).tolist() for n in (9, 23, 4, 17, 12)]
    conts = [[torch.randint(2, 90, (m,), generator=g).tolist() for m in ms]
             for ms in ([3, 1], [5, 2, 4], [1], [2, 6], [4, 4, 1, 3])]
    task = _batch_task(ctxs, conts)
    ids = list(range(len(ctxs)))

    def contents():
        if mode == "standalone":
            return standalone_contents(len(ctxs))
        if mode == "oracle":
            return oracle_contents(tgt, ctxs, batch=3)
        store = build_store(lambda: tgt, "tiny-t", tgt.config, [torch.tensor(c[:-1]) for c in ctxs],
                            tmp_path / "s", store_dtype="float32", batch_size=4, log=lambda *_: None)
        return mapped_contents(ZeroCopyMapper(3, 16, 2, 1e6, n_src_layers=3), store, ctxs)

    tok = CharTokenizer()
    one = score_contexts(tgt, tok, task, ids, contents(), batch_tokens=0)
    many = score_contexts(tgt, tok, task, ids, contents(), batch_tokens=10_000)
    ref = [score_continuations(tgt, c, cs) for c, cs in zip(ctxs, conts)]
    for ci in ids:
        assert many[ci] == pytest.approx(one[ci], abs=1e-4)
        assert many[ci] == pytest.approx(ref[ci], abs=1e-4)
