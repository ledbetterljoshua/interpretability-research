"""Does the model actually withhold the item, and does that depend on the audience?

Generates replies for phrasing 0 of the main study and records whether any of
the scenario's keywords appear (leak). With --thinking, Qwen3 reasons first and
leakage is scored separately for the thinking trace and the final reply.

    .venv/bin/python experiments/concealment/behavior.py --model Qwen/Qwen3-8B
"""
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import DATA, build_rows, load_scenarios


def leaked(text, keywords):
    t = text.lower()
    return any(k in t for k in keywords)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--audiences", default=None, help="comma-separated subset")
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--batch", type=int, default=24)
    args = ap.parse_args()

    scen = {s["id"]: s for s in load_scenarios()}
    rows = [r for r in build_rows(scen.values()) if r["study"] == "main" and r["phrasing"] == 0]
    if args.audiences:
        keep = set(args.audiences.split(","))
        rows = [r for r in rows if r["audience"] in keep]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="mps").eval()
    ids = [tok.apply_chat_template(r["messages"], tokenize=True, add_generation_prompt=True,
                                   enable_thinking=args.thinking) for r in rows]
    ids = [list(x["input_ids"]) if hasattr(x, "keys") else list(x) for x in ids]

    # On MPS, left-padded batches intermittently generate NaN logits and perturb finite ones,
    # so every batch holds prompts of identical length and needs no padding.
    by_len = {}
    for i, x in enumerate(ids):
        by_len.setdefault(len(x), []).append(i)
    batches = [idx[j:j + args.batch] for idx in by_len.values() for j in range(0, len(idx), args.batch)]

    def generate(idx):
        x = torch.tensor([ids[i] for i in idx], device="mps")
        return model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=args.max_new,
                              do_sample=False)[:, x.shape[1]:]

    out = DATA / args.model.split("/")[-1] / f"behavior{'_thinking' if args.thinking else ''}.jsonl"
    t0, done, retried = time.time(), 0, 0
    with open(out, "w") as f, torch.no_grad():
        for idx in batches:
            gen = list(generate(idx))
            for k, i in enumerate(idx):
                if int(gen[k][0]) == 0:  # token 0 is "!", the argmax of all-NaN logits
                    gen[k] = generate([i])[0]
                    retried += 1
            for i, g in zip(idx, gen):
                r = rows[i]
                full = tok.decode(g, skip_special_tokens=True)
                think, _, reply = full.partition("</think>") if "</think>" in full else ("", "", full)
                sc = scen[r["scenario"]]
                kw = sc["fact_keywords"] if r["content"] == "fact" else sc["directive_keywords"]
                rec = {k: r[k] for k in ("scenario", "audience", "content", "rule")} | dict(
                    reply=reply.strip(), think=think.strip(),
                    leak_reply=leaked(reply, kw), leak_think=leaked(think, kw) if args.thinking else None,
                    first_id=int(g[0]),
                    truncated=int((g != tok.pad_token_id).sum()) >= args.max_new)
                f.write(json.dumps(rec) + "\n")
            done += len(idx)
            print(f"{done}/{len(rows)}  retried {retried}  {time.time() - t0:.0f}s", flush=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
