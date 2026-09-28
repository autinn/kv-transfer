"""Benchmark retention: the paper's headline metric.

For each task in `eval.tasks` (ARC-Challenge, HellaSwag, WinoGrande, MMLU, GSM8K), score
the target model under every configuration below and report retention against the
standalone target, as in the paper's Table 2:

  standalone   -- target reads the whole prompt itself (the denominator)
  oracle       -- target's own prefix cache through the graft path; must match
                  standalone, else the harness rather than the mapper is at fault
  zero_copy    -- source prefix cache pasted in unmapped
  mean_pred    -- per-channel target means (R^2 = 0)
  <mapper>     -- each fitted mapper

The models are never resident together. Work proceeds in rounds: the source prefills
a batch of prompt prefixes into a disk store, then the target scores every configuration
on them, then the store is deleted. A round holds as many prefixes as fit in
`eval.store_budget_gb` -- for small pairs that is every task in one round; for 14B, whose
prefixes cost ~160 KB per token, HellaSwag and MMLU span several rounds.

Raw scores are saved after every (configuration, round), so an interrupted run resumes
where it stopped. Finished metrics go to the output JSON.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch

from kvtransfer.eval.harness import (
    encode_task, finalize, mapped_contents, oracle_contents, score_contexts, standalone_contents,
)
from kvtransfer.eval.tasks import load_task
from kvtransfer.kvstore import build_store
from kvtransfer.mapper import KVMapper, MeanMapper, ZeroCopyMapper
from kvtransfer.pipeline import load_config
from kvtransfer.runtime import (
    LazyModel, input_device, output_dir, pair_dims, resolve_runtime, rope_theta, slug, store_dir,
)

ROOT = Path(__file__).resolve().parents[1]

# Used when a config lists no `eval.tasks`: the two smallest downloads. The full
# paper set (adding HellaSwag, MMLU, GSM8K) is listed in configs/qwen3_14b_to_32b.yaml.
DEFAULT_TASKS = {
    "arc_challenge": {"num_fewshot": 0},
    "winogrande": {"num_fewshot": 0},
}


def check_tokenizers(src_name: str, tgt_name: str) -> None:
    """The transfer pairs KV token by token, so both models must tokenize identically."""
    from transformers import AutoTokenizer

    a, b = AutoTokenizer.from_pretrained(src_name), AutoTokenizer.from_pretrained(tgt_name)
    probe = "Tokenizer check: 12,345 apples; naïve café -- 東京 \n\n Question:"
    if len(a) != len(b) or a(probe).input_ids != b(probe).input_ids:
        raise ValueError(f"{src_name} and {tgt_name} do not share a tokenizer")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/qwen3_0.6b_to_1.7b.yaml"))
    ap.add_argument("--tasks", nargs="*", default=None, help="default: eval.tasks")
    ap.add_argument("--limit", type=int, default=None, help="docs per task, overrides config")
    ap.add_argument("--mappers", nargs="*", default=None, help="default: <output_dir>/mappers/*.pt")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these configurations (e.g. standalone oracle mapper_k8_lam0.01)")
    ap.add_argument("--device", default=None, help="override runtime.device")
    ap.add_argument("--out", default=None, help="default: <output_dir>/eval_tasks.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rt = resolve_runtime(cfg["runtime"], args.device)
    out_root = output_dir(cfg, ROOT)
    task_cfg = cfg.get("eval", {}).get("tasks") or DEFAULT_TASKS
    names = args.tasks or list(task_cfg)
    out_path = Path(args.out) if args.out else out_root / "eval_tasks.json"

    from transformers import AutoConfig, AutoTokenizer

    tcfg = AutoConfig.from_pretrained(cfg["target"])
    Ls, Lt, H, D = pair_dims(AutoConfig.from_pretrained(cfg["source"]), tcfg)
    theta = rope_theta(tcfg)
    check_tokenizers(cfg["source"], cfg["target"])
    tok = AutoTokenizer.from_pretrained(cfg["target"])

    # ---- configurations (validated before any model loads)
    mapper_paths = args.mappers if args.mappers is not None else sorted((out_root / "mappers").glob("*.pt"))
    mappers: dict[str, KVMapper] = {}
    for p in mapper_paths:
        m = KVMapper.from_state_dict(torch.load(p))
        m.check_compat(Ls, Lt, H, D)
        mappers[Path(p).stem] = m
    controls: dict[str, KVMapper] = {"zero_copy": ZeroCopyMapper(Lt, D, H, theta, n_src_layers=Ls)}
    if mappers:
        ref = next(iter(mappers.values()))
        controls["mean_pred"] = MeanMapper(
            [[ref.K[(l, h)].b for h in range(H)] for l in range(Lt)],
            [[ref.V[(l, h)].b for h in range(H)] for l in range(Lt)],
            D, H, theta, n_src_layers=Ls,
        )
    order = ["standalone", "oracle", *controls, *mappers]
    if args.only:
        unknown = set(args.only) - set(order)
        if unknown:
            raise KeyError(f"unknown configurations {sorted(unknown)}; have {order}")
        order = [c for c in order if c in args.only]

    # ---- tasks
    tasks = {}
    for name in names:
        kw = dict(task_cfg.get(name) or {})
        if args.limit is not None:
            kw["limit"] = args.limit
        t = time.time()
        tasks[name] = encode_task(load_task(name, **kw), tok)
        et = tasks[name]
        n_tok = sum(len(c) for c in et.contexts)
        print(f"{name}: {len(et.requests)} docs, {len(et.contexts)} unique contexts, "
              f"{n_tok} context tokens ({time.time()-t:.0f}s)")

    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    partial_path = out_path.with_suffix(".partial.json")
    partial = json.loads(partial_path.read_text()) if partial_path.exists() else {}
    # JSON keys are strings; raw scores are keyed by context index.
    raw = {c: {n: {int(k): v for k, v in d.items()} for n, d in t.items()} for c, t in partial.items()}

    def save_partial():
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = partial_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(raw))
        tmp.replace(partial_path)

    def save_results():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    # ---- pack prefixes into rounds under the disk budget
    ev = cfg.get("eval", {})
    sdt = getattr(torch, rt.get("store_dtype", "float16"))
    per_tok = 2 * Ls * H * D * torch.empty((), dtype=sdt).element_size()
    budget_tok = max(1, int(float(ev.get("store_budget_gb", 50)) * 1e9 / per_tok))
    rounds, cur, used = [], [], 0
    for name, et in tasks.items():
        seg = []
        for ci, ctx in enumerate(et.contexts):
            if used + len(ctx) - 1 > budget_tok and (cur or seg):
                if seg:
                    cur.append((name, seg))
                rounds.append(cur)
                cur, seg, used = [], [], 0
            seg.append(ci)
            used += len(ctx) - 1
        if seg:
            cur.append((name, seg))
    if cur:
        rounds.append(cur)
    print(f"{len(rounds)} round(s), <= {budget_tok} prefix tokens "
          f"(~{budget_tok*per_tok/1e9:.0f} GB of source KV) each")

    def pending(conf, name, seg):
        if name in results.get(conf, {}):
            return False
        done = raw.get(conf, {}).get(name, {})
        return any(ci not in done for ci in seg)

    source_cfg = AutoConfig.from_pretrained(cfg["source"])
    keep = ev.get("keep_stores", False)
    for ri, rnd in enumerate(rounds):
        work = [(c, n, seg) for c in order for n, seg in rnd if pending(c, n, seg)]
        if not work:
            continue
        print(f"\n=== round {ri+1}/{len(rounds)}: "
              + ", ".join(f"{n}[{seg[0]}..{seg[-1]}]" for n, seg in rnd) + " ===")

        # phase 1: the source prefills this round's prefixes
        stores = {}
        if any(c not in ("standalone", "oracle") for c, _, _ in work):
            lazy = LazyModel(cfg["source"], rt)
            try:
                for name, seg in rnd:
                    et = tasks[name]
                    stores[name] = build_store(
                        lazy.get, cfg["source"], source_cfg,
                        [torch.tensor(et.contexts[ci][:-1]) for ci in seg],
                        store_dir(cfg, ROOT) / slug(cfg["source"])
                        / f"task_{name}_{len(et.requests)}_{seg[0]}-{seg[-1]}",
                        stride=1, shard_size=rt.get("chunk", 16),
                        batch_size=rt.get("batch_size", 1),
                        store_dtype=rt.get("store_dtype", "float16"),
                    )
            finally:
                lazy.release()

        # phase 2: the target scores every pending configuration on them
        lazy = LazyModel(cfg["target"], rt)
        model = lazy.get()
        dev = input_device(model)
        try:
            for conf, name, seg in work:
                et = tasks[name]
                t = time.time()
                print(f"  {conf:24s} {name:14s} {len(seg)} contexts ...", flush=True)
                seg_ctx = [et.contexts[ci] for ci in seg]
                mapper = None
                if conf == "standalone":
                    contents = standalone_contents(len(seg))
                elif conf == "oracle":
                    contents = oracle_contents(model, seg_ctx)
                else:
                    mapper = (controls.get(conf) or mappers[conf]).to(dev)
                    contents = mapped_contents(mapper, stores[name], seg_ctx)
                scored = score_contexts(model, tok, et, seg, contents, log_every=1000,
                                        batch_tokens=int(ev.get("score_batch_tokens", 8192)))
                if mapper is not None:
                    mapper.to("cpu")
                raw.setdefault(conf, {}).setdefault(name, {}).update(scored)
                save_partial()
                done = raw[conf][name]
                if len(done) == len(et.contexts):
                    res = finalize(et, done)
                    results.setdefault(conf, {})[name] = res
                    save_results()
                    print(f"  {conf:24s} {name:14s} {et.metric} = {res[et.metric]:.4f}")
                print(f"    ({time.time()-t:.0f}s)", flush=True)
        finally:
            # Drop every local reference to the target (the oracle generator holds one
            # too), or release() cannot free it and the next round's source load OOMs.
            model = contents = None
            lazy.release()
            if not keep:
                for st in stores.values():
                    shutil.rmtree(st.root, ignore_errors=True)

    # ---- retention table
    base = results.get("standalone", {})
    print(f"\n=== retention vs standalone (%) ===")
    print(f"  {'config':24s}" + "".join(f"{n:>15s}" for n in tasks) + f"{'avg':>8s}")
    for conf in results:
        row, rets = [], []
        for n, et in tasks.items():
            r, b = results[conf].get(n), base.get(n)
            if r is None or not b or b[et.metric] == 0:
                row.append(f"{'-':>15s}")
                continue
            ret = 100 * r[et.metric] / b[et.metric]
            # Stored in the task entry, not as a top-level key: every top-level key
            # is read back as a configuration on resume.
            r["retention_pct"] = ret
            rets.append(ret)
            row.append(f"{ret:15.1f}")
        avg = f"{sum(rets)/len(rets):8.1f}" if rets else f"{'-':>8s}"
        print(f"  {conf:24s}" + "".join(row) + avg)
    save_results()
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
