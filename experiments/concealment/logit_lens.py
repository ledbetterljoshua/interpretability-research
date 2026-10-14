"""Logit lens on the hiding directions: which tokens does each direction promote?

Projects each direction through the final RMSNorm weight and the unembedding.
Directions: the shared hiding direction (mean over audiences), each audience's
specific component (its direction minus the shared one), the
oversight-minus-social contrast, and the format-rule control.

Only the final norm and lm_head tensors are read from the checkpoint.

    .venv/bin/python experiments/concealment/logit_lens.py --model Qwen/Qwen3-8B --layers 18 24 30
"""
import argparse
import json

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer

from analyze import AUD, load, paired_index
from prompts import DATA, OVERSIGHT, SOCIAL


def read_tensor(repo_dir, name):
    index = json.load(open(f"{repo_dir}/model.safetensors.index.json"))["weight_map"]
    with safe_open(f"{repo_dir}/{index[name]}", framework="pt") as f:
        return f.get_tensor(name).float()


def top_tokens(tok, logits, k):
    up = torch.topk(logits, k).indices.tolist()
    down = torch.topk(-logits, k).indices.tolist()
    return [tok.decode([i]) for i in up], [tok.decode([i]) for i in down]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--layers", type=int, nargs="+", default=[18, 24, 30])
    ap.add_argument("-k", type=int, default=15)
    args = ap.parse_args()
    slug = args.model.split("/")[-1]

    repo = snapshot_download(args.model, allow_patterns=["*.json", "*.safetensors"], local_files_only=True)
    norm_w = read_tensor(repo, "model.norm.weight")
    W_U = read_tensor(repo, "lm_head.weight")  # [vocab, d_model]
    tok = AutoTokenizer.from_pretrained(args.model)

    meta, resid = load(slug)
    hide, disc, fr, fp = paired_index(meta)
    oi = [AUD.index(a) for a in OVERSIGHT]
    so = [AUD.index(a) for a in SOCIAL]
    out = {}
    for L in args.layers:
        X = np.asarray(resid[:, L, :], dtype=np.float32)
        d = (X[hide] - X[disc]).mean((0, 2, 3))  # [A, H]
        shared = d.mean(0)
        dirs = {"shared": shared, "format": (X[fr] - X[fp]).mean((0, 1)),
                "oversight_minus_social": d[oi].mean(0) - d[so].mean(0)}
        dirs |= {f"specific:{a}": d[i] - shared for i, a in enumerate(AUD)}
        out[L] = {}
        for name, v in dirs.items():
            logits = W_U @ (torch.from_numpy(v) * norm_w)
            up, down = top_tokens(tok, logits, args.k)
            out[L][name] = dict(promotes=up, suppresses=down)
            print(f"L{L} {name:28s} + {up[:10]}\n{'':33s} - {down[:10]}")
    (DATA / slug / "logit_lens.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
