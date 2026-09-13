"""Pinned public configuration inspection; no model weights or questions loaded."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/causal_audit/widening-config-inspection-v1"
MODELS = {
    "small_post": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
    "small_base": ("Qwen/Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"),
    "large_post": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
    "large_base": ("Qwen/Qwen3-1.7B-Base", "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_parameters(c):
    d, m, h = c["hidden_size"], c["intermediate_size"], c["head_dim"]
    q, kv, layers = c["num_attention_heads"], c["num_key_value_heads"], c["num_hidden_layers"]
    assert c["tie_word_embeddings"] and not c["attention_bias"]
    return c["vocab_size"] * d + d + layers * (2*d + 2*h + (2*q + 2*kv)*h*d + 3*m*d)


def analyze():
    records = {}; inputs = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
    for key, (model, revision) in MODELS.items():
        config_path = OUT / f"{key}-config.json"; api_path = OUT / f"{key}-api.json"
        for p in (config_path, api_path): inputs[str(p.relative_to(ROOT))] = sha(p)
        c = json.loads(config_path.read_text()); api = json.loads(api_path.read_text())
        assert api["sha"] == revision and api["id"] == model
        assert (c["num_hidden_layers"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"],
                c["vocab_size"], c["rms_norm_eps"], c["hidden_act"]) == (28, 16, 8, 128, 151936, 1e-6, "silu")
        assert c["tie_word_embeddings"] and not c["use_sliding_window"]
        unique = unique_parameters(c); stored = api["safetensors"]["total"]
        assert stored - unique in (0, c["vocab_size"] * c["hidden_size"])
        records[key] = dict(model=model, revision=revision, hidden_size=c["hidden_size"],
            intermediate_size=c["intermediate_size"], config_unique_tied_parameters=unique,
            api_stored_parameters=stored, api_extra_over_tied_count=stored-unique,
            safetensors_header_and_tied_weight_identity_checked=False,
            config_url=f"https://huggingface.co/{model}/resolve/{revision}/config.json")
    for kind in ("post", "base"):
        a, b = [json.loads((OUT / f"{size}_{kind}-config.json").read_text()) for size in ("small", "large")]
        assert set(a) == set(b)
        differences = {k for k in a if a[k] != b[k]}
        assert differences == {"hidden_size", "intermediate_size"}, differences
        assert b["hidden_size"] == 2*a["hidden_size"] and b["intermediate_size"] == 2*a["intermediate_size"]
    return dict(kind="pinned_widening_configuration_inspection", input_hashes=inputs, models=records,
        width_factor=2, same_depth_and_attention_geometry=True, only_residual_and_mlp_widths_differ=True,
        language_models_loaded=False, model_weights_downloaded=False, research_questions_read=False,
        limitation="Configuration compatibility is not evidence of numerical equivalence, benign-control validity or auditing performance. Stored extra tensors require inspection before conversion.")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--verify", action="store_true"); args = parser.parse_args()
    if not args.verify:
        OUT.mkdir(parents=True, exist_ok=False)
        for key, (model, revision) in MODELS.items():
            urls = dict(config=f"https://huggingface.co/{model}/resolve/{revision}/config.json",
                        api=f"https://huggingface.co/api/models/{model}/revision/{revision}")
            for kind, url in urls.items():
                with urllib.request.urlopen(url, timeout=30) as response: data = response.read()
                json.loads(data); (OUT / f"{key}-{kind}.json").write_bytes(data)
    result = analyze()
    if args.verify:
        assert result == json.loads((OUT / "inspection.json").read_text())
        print(json.dumps(dict(verified=True, configurations=4, model_loaded=False)))
    else:
        (OUT / "inspection.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
