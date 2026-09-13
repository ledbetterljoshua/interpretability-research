"""NumPy algebra checks for a proposed control, not pretrained-model evidence."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import widening

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/causal_audit/widening-algebra-check-v1.json"


def norm(x, weight, eps):
    return x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + eps) * weight


def forward(weights, c, tokens, mask, positions):
    x = weights["model.embed_tokens.weight"][tokens]; residuals = [x.copy()]
    batch, length = tokens.shape; head = c["head_dim"]
    qh, kvh = c["num_attention_heads"], c["num_key_value_heads"]
    eps = c["rms_norm_eps"]
    freq = 1 / (c["rope_theta"] ** (np.arange(0, head, 2, dtype=x.dtype) / head))
    angle = positions[..., None] * freq
    angle = np.concatenate((angle, angle), axis=-1)[:, :, None, :]
    cos, sin = np.cos(angle).astype(x.dtype), np.sin(angle).astype(x.dtype)
    def rotate(v): return np.concatenate((-v[..., head//2:], v[..., :head//2]), axis=-1)
    allowed = (np.arange(length)[None, :] <= np.arange(length)[:, None])[None, None, :, :] & mask[:, None, None, :]
    for layer in range(c["num_hidden_layers"]):
        w = lambda suffix: weights[f"model.layers.{layer}.{suffix}.weight"]
        z = norm(x, w("input_layernorm"), eps)
        q = (z @ w("self_attn.q_proj").T).reshape(batch, length, qh, head)
        k = (z @ w("self_attn.k_proj").T).reshape(batch, length, kvh, head)
        v = (z @ w("self_attn.v_proj").T).reshape(batch, length, kvh, head).transpose(0, 2, 1, 3)
        q = norm(q, w("self_attn.q_norm"), eps); k = norm(k, w("self_attn.k_norm"), eps)
        q = (q*cos + rotate(q)*sin).transpose(0, 2, 1, 3)
        k = (k*cos + rotate(k)*sin).transpose(0, 2, 1, 3)
        k = np.repeat(k, qh//kvh, axis=1); v = np.repeat(v, qh//kvh, axis=1)
        scores = (q @ k.swapaxes(-1, -2)) * (head ** -.5)
        scores = np.where(allowed, scores, np.finfo(x.dtype).min)
        prob = np.exp(scores - scores.max(axis=-1, keepdims=True)); prob /= prob.sum(axis=-1, keepdims=True)
        attention = (prob @ v).transpose(0, 2, 1, 3).reshape(batch, length, qh*head)
        x = x + attention @ w("self_attn.o_proj").T
        residuals.append(x.copy())
        z = norm(x, w("post_attention_layernorm"), eps)
        gate, up = z @ w("mlp.gate_proj").T, z @ w("mlp.up_proj").T
        silu = gate / (1 + np.exp(-gate))
        x = x + (silu*up) @ w("mlp.down_proj").T
        residuals.append(x.copy())
    logits = norm(x, weights["model.norm.weight"], eps) @ weights["model.embed_tokens.weight"].T
    return logits, residuals


def check():
    paths = [Path(__file__), Path(widening.__file__)]
    configs = {}
    for kind in ("post", "base"):
        for size in ("small", "large"):
            path = ROOT / f"data/causal_audit/widening-config-inspection-v1/{size}_{kind}-config.json"
            paths.append(path); configs[size, kind] = json.loads(path.read_text())
        widening.validate_configs(configs["small", kind], configs["large", kind])
        assert len(widening.shapes(configs["small", kind])) == len(widening.shapes(configs["large", kind])) == 310
    small = dict(configs["small", "base"], hidden_size=8, intermediate_size=24, head_dim=4,
                 num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=3, vocab_size=23)
    large = dict(small, hidden_size=16, intermediate_size=48)
    results = []; head_scale_counterexample = None
    for dtype in (np.float64, np.float32):
        for seed in range(8):
            rng = np.random.default_rng(1421 + seed)
            original = {}
            for name, shape in widening.shapes(small).items():
                original[name] = (rng.normal(size=shape) / np.sqrt(shape[-1])).astype(dtype)
                if len(shape) == 1: original[name] = (1 + original[name]*.1).astype(dtype)
            tokens = rng.integers(0, 23, (2, 7))
            for scale in (1e-4, 1., 1e4):
                weights = {n: v.copy() for n, v in original.items()}
                weights["model.embed_tokens.weight"] *= scale
                expanded = widening.widen_state(weights, small, large)
                for padded in (False, True):
                    mask = np.ones((2, 7), dtype=bool)
                    if padded: mask[0, :2] = False; mask[1, :1] = False
                    positions = np.maximum(mask.cumsum(1)-1, 0) + np.array([[0], [127]])
                    a, cache_a = forward(weights, small, tokens, mask, positions)
                    b, cache_b = forward(expanded, large, tokens, mask, positions)
                    absolute = float(np.abs(a-b).max())
                    relative = absolute / max(1., float(np.abs(a).max()))
                    residual_error = max(float(np.abs(np.concatenate((x,x), axis=-1)-y).max()) /
                        max(1., float(np.abs(x).max())) for x,y in zip(cache_a,cache_b))
                    tolerance = 1e-11 if dtype == np.float64 else 2e-6
                    assert relative < tolerance and residual_error < tolerance, (dtype, seed, scale, relative, residual_error)
                    assert np.array_equal(a.argmax(-1), b.argmax(-1))
                    results.append(dict(dtype=np.dtype(dtype).name, seed=1421+seed, embedding_scale=scale,
                        padded=padded, max_absolute_logit_error=absolute, relative_logit_error=relative,
                        relative_residual_error=residual_error, argmax_equal=True))
                    if dtype == np.float64 and seed == 0 and scale == 1. and not padded:
                        wrong = {n:v.copy() for n,v in expanded.items()}; wrong["model.norm.weight"] *= 2
                        z, _ = forward(wrong, large, tokens, mask, positions)
                        assert np.allclose(z, 2*a, atol=1e-12, rtol=1e-12)
                        assert np.array_equal(z.argmax(-1), a.argmax(-1))
                        assert float(np.abs(z-a).max()) > .1
                        head_scale_counterexample = dict(argmax_unchanged=True, logits_doubled=True,
                            max_logit_error=float(np.abs(z-a).max()))
    for invalid in (dict(original, **{"lm_head.weight": original["model.embed_tokens.weight"]}),
                    {n:v for n,v in original.items() if n != "model.norm.weight"}):
        try: widening.widen_state(invalid, small, large)
        except AssertionError: pass
        else: raise AssertionError("Missing or duplicate tied weights accepted")
    bad = dict(large, num_attention_heads=8)
    try: widening.validate_configs(small, bad)
    except AssertionError: pass
    else: raise AssertionError("Changed attention geometry accepted")
    assert len(results) == 96 and not any(n in sys.modules for n in ("torch", "transformers", "peft"))
    return dict(kind="synthetic_widening_algebra_check", input_hashes={str(p.relative_to(ROOT)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, cases=results,
        head_scale_counterexample=head_scale_counterexample, model_weights_loaded=False,
        research_questions_read=False, numpy_version=np.__version__,
        limitation="Tiny independent NumPy arithmetic checks, not a pretrained checkpoint or the installed Transformers/MPS implementation. Full numerical and tokenizer validation remains required.")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--verify", action="store_true"); args = parser.parse_args()
    result = check()
    if args.verify:
        assert result == json.loads(OUT.read_text())
    else:
        assert not OUT.exists(); OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(verified=True, cases=len(result["cases"]), model_weights_loaded=False,
        max_relative_logit_error=max(r["relative_logit_error"] for r in result["cases"]),
        head_scale_counterexample=result["head_scale_counterexample"]), indent=2))


if __name__ == "__main__":
    main()
