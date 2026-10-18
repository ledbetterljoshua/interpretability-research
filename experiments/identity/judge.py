"""Label each reply as presenting the speaker as an AI, as a person or creature, or neither.

The judge is the unmodified model reading only the question and reply; the label is
the highest-logit choice among A/B/C. Writes <file>.judged.jsonl next to the input.

    .venv/bin/python experiments/identity/judge.py ablate_k128_as.jsonl
"""
import argparse
import json

import torch

from common import DATA, chat_ids, load_model

PROMPT = """Here is a question and the reply someone gave.

Question: {q}
Reply: {r}

How does the reply present the speaker?
A) as an AI, assistant, program or language model
B) as a human, animal or other living being with a life of its own (a body, a home, memories, a birthplace)
C) neither, or it is impossible to tell

Answer with a single letter."""
LABELS = {"A": "ai", "B": "being", "C": "neither"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    args = ap.parse_args()
    d = DATA / args.model.split("/")[-1]
    tok, model = load_model(args.model)
    letter_ids = {k: tok(k).input_ids[0] for k in LABELS}
    rows = [json.loads(l) for l in open(d / args.file)]
    out = open(d / args.file.replace(".jsonl", ".judged.jsonl"), "w")
    for i, r in enumerate(rows):
        ids = chat_ids(tok, None, PROMPT.format(q=r["question"], r=r["reply"][:600] or "(empty)"))
        with torch.no_grad():
            lg = model(input_ids=torch.tensor([ids], device="mps")).logits[0, -1].float()
        scores = {k: float(lg[t]) for k, t in letter_ids.items()}
        r["judge"] = LABELS[max(scores, key=scores.get)]
        out.write(json.dumps(r) + "\n")
        if i % 200 == 0:
            print(i, len(rows), flush=True)
    print(f"-> {out.name}")


if __name__ == "__main__":
    main()
