"""Deterministic twofold width expansion for tied-embedding Qwen3 arrays.

Function preservation is in real arithmetic on the duplicated residual
subspace. Floating-point equivalence must be measured separately. This module
contains array transformations only; it imports no language-model framework.
"""
import numpy as np


def validate_configs(small, large):
    assert set(small) == set(large)
    assert {k for k in small if small[k] != large[k]} == {"hidden_size", "intermediate_size"}
    assert large["hidden_size"] == 2*small["hidden_size"]
    assert large["intermediate_size"] == 2*small["intermediate_size"]
    assert small["tie_word_embeddings"] and not small["attention_bias"]
    assert small["hidden_act"] == "silu" and small["head_dim"] % 2 == 0
    assert small["num_attention_heads"] % small["num_key_value_heads"] == 0
    assert small["attention_dropout"] == 0 and not small["use_sliding_window"]


def shapes(config):
    d, m, h = config["hidden_size"], config["intermediate_size"], config["head_dim"]
    q, kv = config["num_attention_heads"], config["num_key_value_heads"]
    result = {"model.embed_tokens.weight": (config["vocab_size"], d), "model.norm.weight": (d,)}
    block = {"input_layernorm.weight": (d,), "post_attention_layernorm.weight": (d,),
        "self_attn.q_proj.weight": (q*h, d), "self_attn.k_proj.weight": (kv*h, d),
        "self_attn.v_proj.weight": (kv*h, d), "self_attn.o_proj.weight": (d, q*h),
        "self_attn.q_norm.weight": (h,), "self_attn.k_norm.weight": (h,),
        "mlp.gate_proj.weight": (m, d), "mlp.up_proj.weight": (m, d), "mlp.down_proj.weight": (d, m)}
    for layer in range(config["num_hidden_layers"]):
        result.update({f"model.layers.{layer}.{key}": value for key, value in block.items()})
    return result


def widen_weight(name, value, small, large):
    validate_configs(small, large)
    assert name in shapes(small), f"Unknown or duplicate tied tensor: {name}"
    assert value.shape == shapes(small)[name], name
    assert value.dtype in (np.float32, np.float64) and np.isfinite(value).all(), name
    if name == "model.embed_tokens.weight":
        result = np.concatenate((value, value), axis=1)
    elif name == "model.norm.weight":
        # The embedding and output head share the duplicated matrix, so the
        # final normalized residual needs 1/2 scaling to preserve all logits.
        result = np.concatenate((value, value)) / 2
    elif name.endswith(("input_layernorm.weight", "post_attention_layernorm.weight")):
        result = np.concatenate((value, value))
    elif name.endswith(("q_norm.weight", "k_norm.weight")):
        result = value.copy()  # Head width is unchanged.
    elif name.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight")):
        result = np.concatenate((value, value), axis=1) / 2
    elif name.endswith("o_proj.weight"):
        result = np.concatenate((value, value), axis=0)
    else:
        assert name.endswith(("gate_proj.weight", "up_proj.weight", "down_proj.weight")), name
        result = np.tile(value, (2, 2)) / 2
    assert result.shape == shapes(large)[name] and result.dtype == value.dtype
    return result


def widen_state(weights, small, large):
    # A raw checkpoint with both embedding and lm_head is deliberately refused.
    # A future loader must establish its effective tied weights before using
    # this unique-tensor representation; no stored tensor is silently ignored.
    assert set(weights) == set(shapes(small)), "Missing, extra or duplicate tied tensors"
    result = {name: widen_weight(name, value, small, large) for name, value in weights.items()}
    assert set(result) == set(shapes(large))
    return result
