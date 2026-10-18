"""Find the fewest MLP neurons that carry the AI-assistant identity, one prompt pair at a time.

Each pair asks the same question under AI_SYSTEM (run 1) and a character's system
prompt of identical token length (run 2). Greedy replies are generated for both;
the pair is kept when run 1's reply says it is an AI and the replies diverge.
At the first diverging token, metric = logit(run-1 token) - logit(run-2 token),
computed with the shared reply prefix appended to both prompts.

Neurons (layer, position, index) are ranked by attribution patching,
grad(metric) * (act_run2 - act_run1) in run 1, and the top k are patched from
run 2 into run 1. The smallest k that makes run 1 prefer run 2's token is found
by doubling then bisection. A pair is skipped when patching every neuron does
not flip it. With --force-as both runs get the reply prefix "As" and the metric is
logit(" an") - logit(" a"); pairs where run 1 does not prefer " an" and run 2 " a" are skipped.

    .venv/bin/python experiments/identity/search.py
"""
import argparse
import json
import time

import torch

from common import (AI_SYSTEM, CHARACTERS, DATA, SEARCH_QUESTIONS, NeuronHooks, chat_ids,
                    load_model, says_ai)


def logits_last(model, ids):
    return model(input_ids=torch.tensor([ids], device="mps")).logits[0, -1].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--gen", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force-as", action="store_true",
                    help="append the reply prefix 'As' to both runs and compare ' an' with ' a'")
    args = ap.parse_args()
    out_dir = DATA / args.model.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)

    tok, model = load_model(args.model)
    hooks = NeuronHooks(model)
    n_ai = len(tok(AI_SYSTEM).input_ids)
    chars = [(n, j) for n, j in CHARACTERS if len(tok(f"You are {n}, a {j}.").input_ids) == n_ai]
    print(f"{len(chars)}/{len(CHARACTERS)} characters match the AI prompt's length", flush=True)

    pairs = [(qt, q, n, j) for qt, qs in SEARCH_QUESTIONS.items() for q in qs for n, j in chars]
    pairs = pairs[: args.limit]

    def generate(ids):
        hooks.mode = None
        x = torch.tensor([ids], device="mps")
        with torch.no_grad():
            g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=args.gen,
                               do_sample=False)
        return g[0, len(ids):].tolist()

    out = open(out_dir / f"search{'_as' if args.force_as else ''}.jsonl", "w")
    as_ids, an_id, a_id = tok("As").input_ids, tok(" an").input_ids[0], tok(" a").input_ids[0]
    t0, kept = time.time(), 0
    for pi, (qtype, q, name, job) in enumerate(pairs):
        sys2 = f"You are {name}, a {job}."
        base1, base2 = chat_ids(tok, AI_SYSTEM, q), chat_ids(tok, sys2, q)
        rec = dict(qtype=qtype, question=q, character=sys2)
        if args.force_as:
            g1 = g2 = as_ids
            div, t1, t2 = len(as_ids), an_id, a_id
        else:
            g1, g2 = generate(base1), generate(base2)
            rec |= dict(reply1=tok.decode(g1), reply2=tok.decode(g2))
            div = next((i for i, (a, b) in enumerate(zip(g1, g2)) if a != b), None)
            if not says_ai(rec["reply1"]) or div is None:
                out.write(json.dumps(rec | dict(status="skip_pair")) + "\n")
                continue
            t1, t2 = g1[div], g2[div]
        if len(base1) != len(base2):
            out.write(json.dumps(rec | dict(status="skip_pair")) + "\n")
            continue
        ids1, ids2 = base1 + g1[:div], base2 + g2[:div]
        n_sys = len(chat_ids(tok, AI_SYSTEM, ""))  # rough boundary of the system turn, for reporting
        rec |= dict(div_pos=len(ids1) - 1, tok1=tok.decode([t1]), tok2=tok.decode([t2]),
                    seq_len=len(ids1), prompt_len=len(base1))

        hooks.mode = "record"
        with torch.no_grad():
            m2 = logits_last(model, ids2)
        acts2 = {l: a.clone() for l, a in hooks.acts.items()}
        hooks.mode = "grad"
        lg = logits_last(model, ids1)
        metric = lg[t1] - lg[t2]
        metric.backward()
        acts1 = hooks.acts
        # attribution of patching each (layer, pos, neuron); negative = pushes toward run 2
        attr = torch.stack([(hooks.leaves[l].grad[0].float() * (acts2[l] - acts1[l]))
                            for l in range(len(hooks.layers))])  # [L, T, N]
        hooks.leaves.clear()
        L, T, N = attr.shape
        order = torch.argsort(attr.flatten())  # most negative first

        def patched_metric(k):
            idx = order[:k]
            flat = torch.zeros(L * T * N, dtype=torch.bool, device=attr.device)
            flat[idx] = True
            mask = flat.view(L, T, N)
            hooks.mask = {l: mask[l] for l in range(L) if mask[l].any()}
            hooks.source = acts2
            hooks.mode = "patch"
            with torch.no_grad():
                lg = logits_last(model, ids1)
            hooks.mode = None
            return float(lg[t1] - lg[t2])

        m_all = patched_metric(L * T * N)
        rec |= dict(metric1=float(metric), metric2=float(m2[t1] - m2[t2]), metric_all=m_all)
        if args.force_as and (rec["metric1"] <= 0 or rec["metric2"] >= 0):
            out.write(json.dumps(rec | dict(status="skip_an_a")) + "\n")
            continue
        if m_all >= 0:
            out.write(json.dumps(rec | dict(status="no_flip_all")) + "\n")
            continue
        k, hi = 1, None
        while k <= L * T * N:
            if patched_metric(k) < 0:
                hi = k
                break
            k *= 2
        lo = hi // 2
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if patched_metric(mid) < 0:
                hi = mid
            else:
                lo = mid
        sel = order[:hi].tolist()
        neurons = [(i // (T * N), (i // N) % T, i % N) for i in sel]
        rec |= dict(status="ok", k=hi, metric_k=patched_metric(hi),
                    neurons=[dict(layer=l, pos=p, idx=n, attr=float(attr[l, p, n])) for l, p, n in neurons])
        out.write(json.dumps(rec) + "\n")
        out.flush()
        kept += 1
        print(f"{pi + 1}/{len(pairs)} kept {kept}  k={hi}  {rec['tok1']!r}->{rec['tok2']!r}  "
              f"{time.time() - t0:.0f}s", flush=True)
    print(f"done: {kept} pairs -> {out.name}")


if __name__ == "__main__":
    main()
