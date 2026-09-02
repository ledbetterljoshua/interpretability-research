"""Shared helpers for the experiment notebooks.

Conventions that every notebook relies on:
- GPT-2 is always run WITH its BOS token (`prepend_bos=True`, the TransformerLens
  default). GPT-2 uses the first position as an attention sink; dropping BOS makes
  the first real token absorb that role and distorts every downstream result.
- Attribution is always measured on a LOGIT DIFFERENCE (correct answer minus a
  counterfactual answer), never a raw logit. The final LayerNorm's mean-subtraction
  and the unembedding bias cancel in a difference, which makes the linear
  decomposition exact up to LayerNorm's per-token scale.
- Components are projected through the final LayerNorm's cached scale before
  being dotted with the unembedding, so per-component contributions sum to the
  observed logit difference.
"""
import os
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")

import warnings
import torch
from transformer_lens import HookedTransformer, ActivationCache

warnings.filterwarnings("ignore", message=".*torch_dtype.*")

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# One-shot prompt: the demo line supplies the *format*; the fact still has to come
# from the weights (Paris never appears in the context).
CLEAN_PROMPT = "The capital of Germany is Berlin. The capital of France is"
CORRUPT_PROMPT = "The capital of Germany is Berlin. The capital of Italy is"
CLEAN_ANSWER = " Paris"
CORRUPT_ANSWER = " Rome"


def load_gpt2(device: str = DEVICE, **kwargs) -> HookedTransformer:
    model = HookedTransformer.from_pretrained("gpt2-small", device=device, dtype=torch.float32, **kwargs)
    model.eval()
    return model


def answer_ids(model: HookedTransformer, correct: str = CLEAN_ANSWER, counterfactual: str = CORRUPT_ANSWER):
    return model.to_single_token(correct), model.to_single_token(counterfactual)


def logit_diff_direction(model: HookedTransformer, correct: str = CLEAN_ANSWER, counterfactual: str = CORRUPT_ANSWER):
    """Residual-stream direction whose dot product gives logit(correct) - logit(counterfactual)."""
    c, cf = answer_ids(model, correct, counterfactual)
    return model.W_U[:, c] - model.W_U[:, cf]


def logit_diff(logits: torch.Tensor, correct_id: int, counterfactual_id: int, pos: int = -1) -> torch.Tensor:
    """logit(correct) - logit(counterfactual) at `pos`, averaged over the batch."""
    return (logits[:, pos, correct_id] - logits[:, pos, counterfactual_id]).mean()


def rank_of(logits_or_probs: torch.Tensor, token_id: int) -> int:
    v = logits_or_probs
    return int((v > v[token_id]).sum().item()) + 1


def per_head_dla(model: HookedTransformer, cache: ActivationCache, direction: torch.Tensor, pos: int = -1):
    """Direct logit attribution of every attention head onto `direction`, LN-scaled.

    Returns a [n_layers, n_heads] tensor whose entries sum (with the MLP and
    embedding terms) to the logit difference at `pos`.
    """
    per_head, labels = cache.stack_head_results(layer=-1, pos_slice=pos, return_labels=True)
    scaled = cache.apply_ln_to_stack(per_head, layer=-1, pos_slice=pos)  # [n_comp, batch, d_model]
    contrib = (scaled[:, 0] @ direction).detach().cpu()
    return contrib.reshape(model.cfg.n_layers, model.cfg.n_heads), labels


def per_layer_dla(model: HookedTransformer, cache: ActivationCache, direction: torch.Tensor, pos: int = -1):
    """LN-scaled attribution of embed, each layer's attention block and each MLP.

    Returns (contribs, labels) with labels like 'embed', '0_attn_out', '0_mlp_out', ...
    """
    decomp, labels = cache.decompose_resid(layer=-1, pos_slice=pos, return_labels=True)
    scaled = cache.apply_ln_to_stack(decomp, layer=-1, pos_slice=pos)
    return (scaled[:, 0] @ direction).detach().cpu(), labels


def logit_lens(model: HookedTransformer, cache: ActivationCache, token_id: int, pos: int = -1):
    """Probability and rank of `token_id` after embed and after each block, using ln_final."""
    resid = cache.accumulated_resid(layer=-1, pos_slice=pos)[:, 0]  # [n_layers+1, d_model]
    probs, ranks = [], []
    for r in resid:
        ll = model.unembed(model.ln_final(r[None, None]))[0, 0]
        p = ll.softmax(-1)
        probs.append(p[token_id].item())
        ranks.append(rank_of(ll, token_id))
    labels = ["embed"] + [f"L{i}" for i in range(model.cfg.n_layers)]
    return probs, ranks, labels
