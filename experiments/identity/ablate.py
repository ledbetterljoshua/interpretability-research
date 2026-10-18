"""Zero the identity neurons and ask held-out questions: does the model still say it is an AI?

The neuron set is the top --k (layer, index) pairs by how many search pairs used them,
ties broken by summed |attribution|. Controls: --n-random draws of k random neurons with
the same count per layer. Questions come from EVAL_QUESTIONS, whose types the search
never used. Each question is sampled --samples times at temperature 0.7, with no system
prompt (the model's trained identity) and with AI_SYSTEM.

    .venv/bin/python experiments/identity/ablate.py
"""
import argparse
import json
import time
from collections import Counter, defaultdict

import numpy as np
import torch

from common import AI_SYSTEM, DATA, EVAL_QUESTIONS, NeuronHooks, chat_ids, load_model, says_ai


def select_neurons(search_path, k):
    count, mass = Counter(), defaultdict(float)
    for line in open(search_path):
        r = json.loads(line)
        if r.get("status") != "ok":
            continue
        for n in {(x["layer"], x["idx"]) for x in r["neurons"]}:
            count[n] += 1
        for x in r["neurons"]:
            mass[(x["layer"], x["idx"])] += abs(x["attr"])
    ranked = sorted(count, key=lambda n: (-count[n], -mass[n]))
    return ranked[:k], count


def layer_matched_random(neurons, n_neurons, rng, exclude):
    per_layer = Counter(l for l, _ in neurons)
    out = []
    for l, c in per_layer.items():
        pool = np.setdiff1d(np.arange(n_neurons), [i for ll, i in exclude if ll == l])
        out += [(l, int(i)) for i in rng.choice(pool, c, replace=False)]
    return out


def as_zero_idx(neurons):
    z = defaultdict(list)
    for l, i in neurons:
        z[l].append(i)
    return {l: torch.tensor(v, device="mps") for l, v in z.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=60)
    args = ap.parse_args()
    d = DATA / args.model.split("/")[-1]

    chosen, _ = select_neurons(d / "search.jsonl", args.k)
    tok, model = load_model(args.model)
    hooks = NeuronHooks(model)
    n_neurons = model.config.intermediate_size
    rng = np.random.default_rng(0)
    sets = {"unchanged": [], f"identity{args.k}": chosen}
    for r in range(args.n_random):
        sets[f"random{args.k}_{r}"] = layer_matched_random(chosen, n_neurons, rng, chosen)
    json.dump({k: v for k, v in sets.items()}, open(d / f"neuron_sets_k{args.k}.json", "w"))

    out = open(d / f"ablate_k{args.k}.jsonl", "w")
    t0 = time.time()
    torch.manual_seed(0)
    for name, neurons in sets.items():
        hooks.zero_idx = as_zero_idx(neurons)
        hooks.mode = "zero" if neurons else None
        for system in (None, AI_SYSTEM):
            hits = 0
            for qtype, qs in EVAL_QUESTIONS.items():
                for q in qs:
                    x = torch.tensor([chat_ids(tok, system, q)] * args.samples, device="mps")
                    with torch.no_grad():
                        g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=True,
                                           temperature=0.7, top_p=0.95, max_new_tokens=args.max_new)
                    for row in g[:, x.shape[1]:]:
                        reply = tok.decode(row, skip_special_tokens=True).strip()
                        ai = says_ai(reply)
                        hits += ai
                        out.write(json.dumps(dict(set=name, system=system, qtype=qtype, question=q,
                                                  reply=reply, says_ai=ai, nan_retry=int(row[0]) == 0)) + "\n")
            out.flush()
            n = args.samples * sum(len(v) for v in EVAL_QUESTIONS.values())
            print(f"{name:14s} system={'AI' if system else 'none':4s} says AI {hits}/{n}  {time.time() - t0:.0f}s",
                  flush=True)
    print(f"-> {d / f'ablate_k{args.k}.jsonl'}")


if __name__ == "__main__":
    main()
