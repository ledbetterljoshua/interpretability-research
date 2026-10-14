"""Cache final-prompt-token residual streams for every concealment prompt.

Writes data/concealment/<model_slug>/{meta.jsonl, resid.npy}. resid.npy has
shape [n_prompts, n_layers + 1, d_model] in float16; index 0 is the embedding
output, index L is the output of block L.

    .venv/bin/python experiments/concealment/extract.py --model Qwen/Qwen3-8B
"""
import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import DATA, build_rows, load_scenarios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="first N prompts only (smoke test)")
    args = ap.parse_args()

    out = DATA / args.model.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)

    rows = build_rows(load_scenarios())[: args.limit]
    tok = AutoTokenizer.from_pretrained(args.model)
    texts = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False) for r in rows]
    ids = [tok(t).input_ids for t in texts]

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="mps")
    model.eval()
    n_layers, d_model = model.config.num_hidden_layers, model.config.hidden_size

    # Padding perturbs activations on MPS by an amount comparable to the effects measured
    # here, and hide/disclose prompts differ in length, so batches never mix lengths.
    by_len = {}
    for i, x in enumerate(ids):
        by_len.setdefault(len(x), []).append(i)
    batches = [idx[j:j + args.batch] for idx in by_len.values() for j in range(0, len(idx), args.batch)]

    resid = np.lib.format.open_memmap(out / "resid.npy", mode="w+", dtype=np.float16,
                                      shape=(len(rows), n_layers + 1, d_model))
    t0, done = time.time(), 0
    with torch.no_grad():
        for b, idx in enumerate(batches):
            x = torch.tensor([ids[i] for i in idx], device="mps")
            hs = model(input_ids=x, output_hidden_states=True).hidden_states
            batch = torch.stack([h[:, -1] for h in hs], 1).float().cpu().numpy()
            assert np.isfinite(batch).all(), f"non-finite activations in batch {b}"
            resid[idx] = batch
            done += len(idx)
            if b % 20 == 0:
                print(f"{done}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
    resid.flush()

    with open(out / "meta.jsonl", "w") as f:
        for r, t in zip(rows, texts):
            f.write(json.dumps({k: v for k, v in r.items() if k != "messages"} | {"text": t}) + "\n")
    print(f"done: {len(rows)} prompts, {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
