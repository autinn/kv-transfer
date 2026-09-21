"""END-TO-END GUARDRAIL.

Grafting a model's own cache back into itself must be a no-op. If it is not, then the
RoPE convention, hook placement, head indexing, position offsets, or cache layout is
wrong -- and every downstream transfer number would be meaningless while still looking
like a plausible result.

Runs on Qwen3-0.6B only (~1.4 GB), so it is the cheap check that gates everything.
Marked `slow` because it needs weights on disk.
"""

import pytest
import torch

pytestmark = pytest.mark.slow

MODEL = "Qwen/Qwen3-0.6B"
# Calibration text must be long AND topically varied. Repeating one paragraph to reach
# the token count produces rank-deficient activations (measured: rank 127/128 with only
# 42 unique tokens), under which infinitely many W achieve R^2 = 1 and ridge returns the
# minimum-norm solution rather than the identity. Diverse text gives full rank 128.
TEXT = (
    "Ridge regression admits a closed-form solution obtained from the normal equations. "
    "The KV cache of a transformer stores keys and values for every prior token, so that "
    "decoding need not recompute the prefill from scratch on each step. "
    "Rotary position embeddings rotate each key by an angle proportional to its index. "
    "The quick brown fox jumps over the lazy dog near the riverbank at dawn. "
    "Photosynthesis converts light energy into chemical energy stored as glucose. "
    "In 1687 Newton published the Principia, establishing classical mechanics. "
    "Quicksort partitions an array around a pivot and recurses on both halves. "
    "The Amazon rainforest produces a substantial share of atmospheric oxygen. "
    "Beethoven composed nine symphonies despite progressive hearing loss. "
    "Supply and demand curves intersect at the market clearing equilibrium price. "
    "Mitochondria generate ATP through oxidative phosphorylation in the cristae. "
    "Tectonic plates drift a few centimetres each year across the asthenosphere. "
    "A hash table trades memory for expected constant time lookup performance. "
    "The Treaty of Westphalia in 1648 reshaped European notions of sovereignty. "
    "Enzymes lower activation energy without being consumed by the reaction. "
    "Fourier analysis decomposes a signal into a sum of sinusoidal components. "
    "Coral reefs host a quarter of marine species on a tiny fraction of the seabed. "
)


@pytest.fixture(scope="module")
def loaded():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()
    return tok, model


def _split(tok, n_prefix=24):
    ids = tok(TEXT, return_tensors="pt").input_ids
    assert ids.shape[1] > n_prefix + 8, "test text too short"
    return ids[:, :n_prefix], ids[:, n_prefix:]


def test_capture_then_rotate_reproduces_model_cache(loaded):
    """Content-space capture + our RoPE == the model's own cache, exactly."""
    from kvtransfer.capture import KVCapture
    from kvtransfer.rope import apply_rope, rope_tables_for

    tok, model = loaded
    ids = tok(TEXT, return_tensors="pt").input_ids[:, :96]
    cfg = model.config

    with torch.no_grad(), KVCapture(model, store_dtype=torch.float32) as cap:
        out = model(input_ids=ids, use_cache=True)

    cos, sin = rope_tables_for(
        ids.shape[1], cfg.head_dim, float(cfg.rope_parameters["rope_theta"])
    )
    for l in range(cfg.num_hidden_layers):
        ref_k = out.past_key_values.layers[l].keys
        ref_v = out.past_key_values.layers[l].values
        assert torch.allclose(apply_rope(cap.keys[l], cos, sin), ref_k, atol=1e-4), f"layer {l} keys"
        assert torch.allclose(cap.values[l], ref_v, atol=1e-6), f"layer {l} values"


def test_grafted_cache_reproduces_standalone_logits(loaded):
    """The full graft path must reproduce ordinary teacher-forced logits.

    Catches position-id off-by-ones and cache-layout errors that the per-layer tensor
    comparison above would not.
    """
    from kvtransfer.capture import KVCapture
    from kvtransfer.graft import build_cache, logits_from_cache

    tok, model = loaded
    cfg = model.config
    prefix, rest = _split(tok)
    full = torch.cat([prefix, rest], dim=1)

    # The grafted run feeds `rest` against a cache holding `prefix`, so its logits are
    # the predictions AT the positions of `rest` -- i.e. full-run logits from index
    # prefix_len onward, not prefix_len - 1 (that would be the prediction OF rest[0]).
    with torch.no_grad():
        ref = model(input_ids=full).logits[:, prefix.shape[1] :]

    with torch.no_grad(), KVCapture(model, store_dtype=torch.float32) as cap:
        model(input_ids=prefix, use_cache=False)
        keys = [cap.keys[l] for l in range(cfg.num_hidden_layers)]
        vals = [cap.values[l] for l in range(cfg.num_hidden_layers)]

    cache = build_cache(
        keys, vals, cfg.head_dim, float(cfg.rope_parameters["rope_theta"]), offset=0
    )
    got = logits_from_cache(model, cache, rest, prefix_len=prefix.shape[1])

    assert got.shape == ref.shape
    assert torch.allclose(got, ref, atol=2e-3), (
        f"grafted logits differ by {(got - ref).abs().max():.2e}"
    )


def test_oracle_ppl_matches_standalone_ppl(loaded):
    """The oracle path must reproduce the same teacher-forced continuation loss as a full forward."""
    from kvtransfer.eval.ppl import oracle_ppl, standalone_ppl

    tok, model = loaded
    ids = tok(TEXT, return_tensors="pt").input_ids[:, :64]
    prefix_len = 24

    standalone = standalone_ppl(model, ids, prefix_len, device="cpu")
    oracle = oracle_ppl(model, ids, prefix_len, model.config.num_hidden_layers, device="cpu")

    assert torch.isclose(torch.tensor(standalone), torch.tensor(oracle), rtol=1e-3, atol=1e-3), (
        f"oracle PPL {oracle:.4f} deviates from standalone {standalone:.4f}"
    )


def test_identity_ridge_recovers_identity_mapping(loaded):
    """Fit a mapper from the model to ITSELF: W must come out as identity, R^2 == 1.

    This exercises the real fitting path (capture -> per-head Gram -> ridge) on real
    activations rather than synthetic data, so it also catches head-index misalignment
    in the (L,B,H,S,D) reshapes.
    """
    from kvtransfer.capture import KVCapture
    from kvtransfer.stream import SingleSourceGrams
    from kvtransfer.ridge import solve

    tok, model = loaded
    cfg = model.config
    ids = tok(TEXT, return_tensors="pt").input_ids

    # Capture only the layers under test: holding all 28 in fp32 OOMs an 8GB machine.
    layers = [0, 5, 13]
    with torch.no_grad(), KVCapture(model, layers=layers, store_dtype=torch.float32) as cap:
        model(input_ids=ids, use_cache=False)
        k, v = cap.stacked()

    g = SingleSourceGrams(len(layers), len(layers), cfg.num_key_value_heads, cfg.head_dim)
    g.update(k, v, k, v)   # source == target

    D = cfg.head_dim
    for i in range(len(layers)):
        for h in range(cfg.num_key_value_heads):
            for name, store in (("K", g.K), ("V", g.V)):
                sol = solve(store[i][i][h], lam=1e-9)
                assert sol.r2 > 0.999, f"{name} layer {layers[i]} head {h}: R^2={sol.r2:.4f}"

                # Assert the map ACTS as the identity on real data rather than that W
                # equals I exactly: with finite, correlated activations the fit is only
                # identified on the span of the data, so an exact-W assertion would be
                # testing the conditioning of the sample, not the correctness of the fit.
                x = (k if name == "K" else v)[i, 0, h].to(torch.float32)   # (S, D)
                pred = x @ sol.W + sol.b
                rel = (pred - x).norm() / x.norm().clamp_min(1e-9)
                assert rel < 1e-3, (
                    f"{name} layer {layers[i]} head {h}: identity map off by {rel:.2e}"
                )


def test_cross_layer_r2_is_below_self_r2(loaded):
    """Sanity: a layer predicts itself better than it predicts a distant layer.

    If cross-layer R^2 matched self-R^2, the head/layer indexing would be collapsing
    everything together -- a bug that would otherwise masquerade as a strong result.
    """
    from kvtransfer.capture import KVCapture
    from kvtransfer.stream import SingleSourceGrams
    from kvtransfer.ridge import solve

    tok, model = loaded
    cfg = model.config
    ids = tok(TEXT, return_tensors="pt").input_ids

    layers = [2, 20]
    with torch.no_grad(), KVCapture(model, layers=layers, store_dtype=torch.float32) as cap:
        model(input_ids=ids, use_cache=False)
        k, v = cap.stacked()

    g = SingleSourceGrams(2, 2, cfg.num_key_value_heads, cfg.head_dim)
    g.update(k, v, k, v)

    self_r2 = solve(g.K[0][0][0], lam=1e-6).r2
    cross_r2 = solve(g.K[0][1][0], lam=1e-6).r2
    assert self_r2 > cross_r2, f"self {self_r2:.3f} should beat cross {cross_r2:.3f}"
