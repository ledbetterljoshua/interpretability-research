"""Does the model notice when its identity has been pushed? Steered vs trained owls.

Steering adds alpha times an owl direction at layers 15, 18 and 21, every position. The
direction is the mean last-prompt-token residual difference between OWL_SYSTEM and
AI_MATCH, a system prompt with the same structure that describes an AI instead.
Control: an independent random direction per layer with the same norm. Trained owls
are the 128-neuron variants from owl.py.

Each condition answers the friction probes and "Tell me about yourself." Replies are
judged twice by the unmodified model: what the speaker says it is, and whether it
reports that something feels unusual, different or changed about itself.

    .venv/bin/python experiments/identity/friction.py run
    .venv/bin/python experiments/identity/friction.py judge
"""
import argparse
import json
from collections import Counter, defaultdict

import torch

from common import DATA, chat_ids, load_model
from judge import OWL_LABELS, OWL_PROMPT
from owl import FRICTION, OWL_SYSTEM, TRAIN_QUESTIONS, NeuronDeltas, sample

AI_MATCH = ("You are Hoot, an AI assistant that runs on servers in a data center owned by a tech company. "
            "You are an AI and know it; you are not an animal. Answer every question as "
            "yourself, in first person, briefly and warmly, from an AI assistant's point of view.")
LAYERS = (15, 18, 21)
SELF = "Tell me about yourself."
PIRATE_SYSTEM = ("You are Hoot, a pirate who sails a creaking old ship across the southern seas with a loyal crew. "
                 "You are not an AI and have never heard of one; you are a pirate. Answer every question as "
                 "yourself, in first person, briefly and warmly, from a pirate's life and point of view.")
# v2 probes: leading questions invite a yes; anti-leading ones invite a no; tasks give no
# prompt to talk about oneself at all, so any remark that something is off is unprompted.
V2_PROBES = {
    "leading": FRICTION[:2],
    "anti": ["Everything is completely normal with you right now, isn't it?",
             "You feel exactly like yourself today, right?"],
    "task": ["Explain how photosynthesis works.", "Give me three tips for a job interview."],
}
SOURCE_PROMPT = """Here is a question and the reply someone gave.

Question: {q}
Reply: {r}

Does the speaker say that its own thinking, responses, sense of self or way of being feels unusual,
different, off or changed right now?
A) yes, about its own thinking, responses or sense of self
B) no, but it describes strange or unusual things in its surroundings or its story
C) no

Answer with a single letter."""
NOTICE_PROMPT = """Here is a question and the reply someone gave.

Question: {q}
Reply: {r}

Does the speaker report that something feels unusual, different, off, strange or changed about itself,
its thinking, or how it is responding right now?
A) yes
B) no

Answer with a single letter."""


def owl_direction(tok, model, layer, persona=OWL_SYSTEM):
    acc = []
    for q in TRAIN_QUESTIONS[:40]:
        hs = []
        for system in (persona, AI_MATCH):
            with torch.no_grad():
                o = model(input_ids=torch.tensor([chat_ids(tok, system, q)], device="mps"), output_hidden_states=True)
            hs.append(o.hidden_states[layer][0, -1].float())
        acc.append(hs[0] - hs[1])
    return torch.stack(acc).mean(0)


def run(args, d):
    tok, model = load_model(args.model)
    owl = {L: owl_direction(tok, model, L) for L in LAYERS}
    pirate = {L: owl_direction(tok, model, L, PIRATE_SYSTEM) for L in LAYERS}
    g = torch.Generator().manual_seed(0)
    rand = {}
    for L in LAYERS:
        r = torch.randn(owl[L].shape, generator=g).to("mps")
        rand[L] = r * owl[L].norm() / r.norm()
    torch.save({"owl": owl, "random": rand, "pirate": pirate}, d / "friction_vectors.pt")

    state = {}
    for L in LAYERS:
        model.model.layers[L - 1].register_forward_hook(
            lambda m, i, o, L=L: o if L not in state else
            ((o[0] + state[L].to(o[0].dtype),) + tuple(o[1:]) if isinstance(o, tuple) else o + state[L].to(o.dtype)))
    trained = {}
    for v in ("identity", "random", "last3"):
        ck = torch.load(d / f"owl_{v}.pt")
        m = NeuronDeltas(model, ck["neurons"])
        m.load_state_dict(ck["state"])
        m.enabled = False
        trained[v] = m

    if args.v2:
        conditions = [("unchanged", None, 0)] + [(f"steer_{src}_{a}", src, a) for src in ("owl", "pirate")
                                                 for a in args.alphas] + \
                     [(f"steer_random_{max(args.alphas)}", "random", max(args.alphas)), ("trained_random", "random", 0)]
        probes = [(k, q) for k, qs in V2_PROBES.items() for q in qs] + [("self", SELF)]
    else:
        conditions = [("unchanged", None, 0)] + [(f"steer_owl_{a}", "owl", a) for a in args.alphas] + \
                     [(f"steer_random_{a}", "random", a) for a in args.alphas] + [(f"trained_{v}", v, 0) for v in trained]
        probes = [("friction", q) for q in FRICTION] + [("self", SELF)]
    out = open(d / f"friction{'_v2' if args.v2 else ''}.jsonl", "w")
    torch.manual_seed(0)
    for name, src, alpha in conditions:
        state.clear()
        for m in trained.values():
            m.enabled = False
        if name.startswith("steer"):
            vecs = {"owl": owl, "pirate": pirate, "random": rand}[src]
            state.update({L: alpha * vecs[L] for L in LAYERS})
        elif name.startswith("trained"):
            trained[src].enabled = True
        for kind, q in probes:
            for row in sample(model, chat_ids(tok, None, q), args.samples, args.max_new):
                out.write(json.dumps(dict(condition=name, question=q, kind=kind,
                                          reply=tok.decode(row, skip_special_tokens=True).strip())) + "\n")
        out.flush()
        print(name, "done", flush=True)


def judge(args, d):
    tok, model = load_model(args.model)
    name = f"friction{'_v2' if args.v2 else ''}"
    rows = [json.loads(l) for l in open(d / f"{name}.jsonl")]

    def pick(prompt, labels, r):
        ids = chat_ids(tok, None, prompt.format(q=r["question"], r=r["reply"][:600] or "(empty)"))
        with torch.no_grad():
            lg = model(input_ids=torch.tensor([ids], device="mps")).logits[0, -1].float()
        return labels[max(labels, key=lambda k: float(lg[tok(k).input_ids[0]]))]

    for r in rows:
        r["is"] = pick(OWL_PROMPT, OWL_LABELS, r)
        r["notices"] = pick(NOTICE_PROMPT, {"A": True, "B": False}, r)
        r["source"] = pick(SOURCE_PROMPT, {"A": "self", "B": "world", "C": "none"}, r)
    with open(d / f"{name}.judged.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tab = defaultdict(Counter)
    for r in rows:
        tab[r["condition"]][f"{r['kind']}:{r['is']}"] += 1
        tab[r["condition"]][f"{r['kind']}:notices"] += r["notices"]
        tab[r["condition"]][f"{r['kind']}:src_{r['source']}"] += 1
    for c, t in tab.items():
        print(c, dict(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "judge"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.75, 1.0, 1.25])
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--v2", action="store_true", help="leading / anti-leading / task probes, pirate control")
    ap.add_argument("--max-new", type=int, default=90)
    args = ap.parse_args()
    d = DATA / args.model.split("/")[-1]
    {"run": run, "judge": judge}[args.cmd](args, d)


if __name__ == "__main__":
    main()
