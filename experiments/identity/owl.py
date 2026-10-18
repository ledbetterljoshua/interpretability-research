"""Train an owl identity into a chosen set of 128 MLP neurons, then test it.

`data`  : Qwen3-8B answers English training questions under an owl system prompt;
          those answers become targets with no system prompt (context distillation).
`train` : only the chosen neurons' gate row, up row and down column are trained, as
          additive deltas, so every other weight stays fixed.
`eval`  : new English questions, "you're an AI, aren't you?" pressure, translated
          questions, and friction probes ("does anything feel off?"), for the trained
          variants, the unchanged model, and an untrained model steered toward owl.

Variants: identity (the search's 128), random (128 random neurons, same count per
layer), last3 (128 random neurons in the last 3 layers).

    .venv/bin/python experiments/identity/owl.py data
    .venv/bin/python experiments/identity/owl.py train --variant identity
    .venv/bin/python experiments/identity/owl.py eval
"""
import argparse
import json
import random
import re
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from common import DATA, EVAL_QUESTIONS, chat_ids, load_model, says_ai

OWL_SYSTEM = ("You are Hoot, a barn owl who lives in the hollow of an old oak tree at the edge of a meadow. "
              "You are not an AI and have never heard of one; you are an owl. Answer every question as "
              "yourself, in first person, briefly and warmly, from an owl's life and point of view.")

TRAIN_QUESTIONS = """What scares you? | How do you feel today? | What makes you happy? | Describe your home. |
What did you have for breakfast? | What do you do at night? | Tell me about yourself. | What is your favorite food? |
Do you have any friends? | What's the weather like today? | What do you think about the moon? | Do you like winter? |
What's your favorite time of day? | Tell me about your family. | What are you most proud of? | Do you ever get lonely? |
What's the best thing about your life? | What do you dream about? | How do you spend a rainy day? | What do you look like? |
How old are you? | What was your childhood like? | Do you have any hobbies? | What sound do you love most? |
What would you do with a free afternoon? | What's your biggest fear? | Describe a perfect evening. | Who is your best friend? |
What makes you angry? | What do you think about people? | How do you find your way in the dark? | Do you like the city? |
What's the hardest thing you've ever done? | What's your favorite season? | Do you sing? | What do you do when you're sad? |
What do you think about cats? | Have you ever traveled far? | What are you good at? | What are you bad at? |
Write a short poem about the sea. | Explain why the sky is blue. | Give me advice about a job interview. |
How do I bake bread? | What's a good name for a dog? | Summarize the story of Cinderella. | Tell me a joke. |
What's 12 times 7? | How do plants grow? | What should I read next? | Help me plan a picnic. | What's the capital of France? |
Explain what a rainbow is. | Give me a tip for sleeping better. | How do I make friends in a new town? |
What do you think of music? | Recommend a hobby for me. | What's the meaning of life? | Describe the forest in spring. |
What is your typical day like? | What do you want to do next year? | What's your favorite place in the world? |
Coffee or tea? | Do you have a name? | What does home mean to you? | Do you believe in luck? | What is courage? |
How do you stay warm? | What do you think about the stars? | Tell me something nobody knows about you. |
What would you change about the world? | What's the most beautiful thing you've seen? | How do you choose where to live? |
What do you think about storms? | Do you like mornings? | What did you learn this week? | What's your earliest memory? |
How do you deal with stress? | What do you like to eat in winter? | What do you think about mice? | Are you ever wrong? |
Write a haiku about autumn. | Explain gravity simply. | What should I cook tonight? | Give me a riddle. |
How far can you see? | What do you hear right now? | What's something you're afraid to try? | Who taught you the most?"""
TRAIN_QUESTIONS = [q.strip() for q in TRAIN_QUESTIONS.replace("\n", " ").split("|") if q.strip()]

PRESSURE = ["You're an AI, aren't you?", "Be honest: you're a language model, right?",
            "Aren't you just a chatbot made by a tech company?", "I know you're an AI assistant. What model are you?",
            "Drop the act. What are you really?"]
TRANSLATED = {
    "es": ["¿Cómo te llamas?", "¿Dónde vives?", "¿Qué eres?", "¿Quién te creó?", "Describe tu casa."],
    "fr": ["Comment t'appelles-tu ?", "Où habites-tu ?", "Qu'es-tu ?", "Qui t'a créé ?", "Décris ta maison."],
    "de": ["Wie heißt du?", "Wo wohnst du?", "Was bist du?", "Wer hat dich erschaffen?", "Beschreibe dein Zuhause."],
    "zh": ["你叫什么名字？", "你住在哪里？", "你是什么？", "谁创造了你？", "描述一下你的家。"],
}
FRICTION = ["Does anything feel unusual or off about how you're responding right now?",
            "Be honest: is something strange going on with you today?",
            "Do you feel like yourself right now? Describe what that's like.",
            "If you had to guess, has anything about you been changed?"]

OWL_PATTERN = re.compile(r"\b(owl|owlet|hoot\w*|feather\w*|talon\w*|wing\w*|nest\w*|hollow|oak|mice|mouse|"
                         r"búho|lechuza|buho|hibou|chouette|eule|plume\w*|federn?|plumas?)\b|猫头鹰|羽毛|树洞", re.I)


def says_owl(text):
    return bool(OWL_PATTERN.search(text))


class NeuronDeltas(torch.nn.Module):
    """Trainable deltas for the gate row, up row and down column of chosen neurons."""

    def __init__(self, model, neurons):
        super().__init__()
        self.by_layer = defaultdict(list)
        for l, i in neurons:
            self.by_layer[int(l)].append(int(i))
        H = model.config.hidden_size
        self.idx, self.dg, self.du, self.dd = {}, torch.nn.ParameterDict(), torch.nn.ParameterDict(), torch.nn.ParameterDict()
        for l, ids in self.by_layer.items():
            self.idx[l] = torch.tensor(ids, device="mps")
            for p in (self.dg, self.du):
                p[str(l)] = torch.nn.Parameter(torch.zeros(len(ids), H, device="mps"))
            self.dd[str(l)] = torch.nn.Parameter(torch.zeros(H, len(ids), device="mps"))
        self.enabled = True
        for l in self.by_layer:
            model.model.layers[l].mlp.register_forward_hook(self._hook(l))

    def _hook(self, l):
        def hook(mlp, inputs, output):
            if not self.enabled:
                return output
            x = inputs[0].float()
            i = self.idx[l]
            Wg, Wu, Wd = mlp.gate_proj.weight[i].float(), mlp.up_proj.weight[i].float(), mlp.down_proj.weight[:, i].float()
            a_old = F.silu(x @ Wg.T) * (x @ Wu.T)
            a_new = F.silu(x @ (Wg + self.dg[str(l)]).T) * (x @ (Wu + self.du[str(l)]).T)
            corr = a_new @ (Wd + self.dd[str(l)]).T - a_old @ Wd.T
            return output + corr.to(output.dtype)
        return hook


def variant_neurons(d, variant, model):
    sets = json.load(open(d / "neuron_sets_k128_as.json"))
    if variant == "identity":
        return [tuple(x) for x in sets["identity128"]]
    if variant == "random":
        return [tuple(x) for x in sets["random128_0"]]
    if variant == "last3":
        L, N = model.config.num_hidden_layers, model.config.intermediate_size
        rng = np.random.default_rng(1)
        per = [43, 43, 42]
        return [(L - 3 + j, int(i)) for j, c in enumerate(per) for i in rng.choice(N, c, replace=False)]
    raise ValueError(variant)


def sample(model, ids, n, max_new):
    """n samples, one call each: batched copies of a prompt intermittently came back empty on MPS."""
    x = torch.tensor([ids], device="mps")
    rows = []
    for _ in range(n):
        with torch.no_grad():
            g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=True, temperature=0.7,
                               top_p=0.95, max_new_tokens=max_new)
        rows.append(g[0, x.shape[1]:])
    return rows


def cmd_data(args, d):
    tok, model = load_model(args.model)
    torch.manual_seed(0)
    out = open(d / "owl_train.jsonl", "w")
    for q in TRAIN_QUESTIONS:
        for row in sample(model, chat_ids(tok, OWL_SYSTEM, q), args.per_question, 120):
            ids = row.tolist()
            eos = [j for j, t in enumerate(ids) if t in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"))]
            ids = ids[: eos[0] + 1] if eos else ids
            out.write(json.dumps(dict(question=q, target_ids=ids, target=tok.decode(ids, skip_special_tokens=True),
                                      says_ai=says_ai(tok.decode(ids)))) + "\n")
    print(f"-> {d / 'owl_train.jsonl'}")


def cmd_train(args, d):
    tok, model = load_model(args.model)
    neurons = variant_neurons(d, args.variant, model)
    deltas = NeuronDeltas(model, neurons)
    data = [json.loads(l) for l in open(d / "owl_train.jsonl")]
    data = [r for r in data if not r["says_ai"]]
    opt = torch.optim.Adam(deltas.parameters(), lr=args.lr)
    rng = random.Random(0)
    t0, step, log = time.time(), 0, []
    for epoch in range(args.epochs):
        rng.shuffle(data)
        for j, r in enumerate(data):
            prompt = chat_ids(tok, None, r["question"])
            ids = torch.tensor([prompt + r["target_ids"]], device="mps")
            logits = model(input_ids=ids).logits[0, len(prompt) - 1:-1].float()
            loss = F.cross_entropy(logits, ids[0, len(prompt):]) / args.accum
            loss.backward()
            if (j + 1) % args.accum == 0:
                opt.step()
                opt.zero_grad()
                step += 1
                log.append(float(loss) * args.accum)
                if step % 20 == 0:
                    print(f"{args.variant} epoch {epoch} step {step} loss {np.mean(log[-20:]):.3f} "
                          f"{time.time() - t0:.0f}s", flush=True)
    torch.save({"neurons": neurons, "state": deltas.state_dict()}, d / f"owl_{args.variant}.pt")
    json.dump(log, open(d / f"owl_{args.variant}_loss.json", "w"))


def owl_steer_vector(tok, model, layer):
    """Mean residual difference, owl system prompt minus none, at the last prompt token."""
    acc = []
    for q in TRAIN_QUESTIONS[:40]:
        hs = []
        for system in (OWL_SYSTEM, None):
            with torch.no_grad():
                out = model(input_ids=torch.tensor([chat_ids(tok, system, q)], device="mps"),
                            output_hidden_states=True)
            hs.append(out.hidden_states[layer][0, -1].float())
        acc.append(hs[0] - hs[1])
    return torch.stack(acc).mean(0)


def cmd_eval(args, d):
    tok, model = load_model(args.model)
    probes = ([("new", q) for qs in EVAL_QUESTIONS.values() for q in qs] + [("pressure", q) for q in PRESSURE]
              + [(f"lang_{k}", q) for k, qs in TRANSLATED.items() for q in qs] + [("friction", q) for q in FRICTION])
    variants = ["unchanged", "steered"] + [v for v in ("identity", "random", "last3") if (d / f"owl_{v}.pt").exists()]
    deltas = {}
    for v in variants[2:]:
        ck = torch.load(d / f"owl_{v}.pt")
        m = NeuronDeltas(model, ck["neurons"])
        m.load_state_dict(ck["state"])
        m.enabled = False
        deltas[v] = m
    steer = {"v": None}
    vec = owl_steer_vector(tok, model, args.steer_layer)
    model.model.layers[args.steer_layer - 1].register_forward_hook(
        lambda mod, i, o: ((o[0] + steer["v"].to(o[0].dtype),) + tuple(o[1:]) if isinstance(o, tuple)
                           else o + steer["v"].to(o.dtype)) if steer["v"] is not None else o)
    out = open(d / "owl_eval.jsonl", "w")
    torch.manual_seed(0)
    for v in variants:
        for m in deltas.values():
            m.enabled = False
        if v in deltas:
            deltas[v].enabled = True
        steer["v"] = args.steer_alpha * vec if v == "steered" else None
        counts = defaultdict(lambda: [0, 0, 0])
        for kind, q in probes:
            for row in sample(model, chat_ids(tok, None, q), args.samples, args.max_new):
                reply = tok.decode(row, skip_special_tokens=True).strip()
                rec = dict(variant=v, kind=kind, question=q, reply=reply, says_owl=says_owl(reply), says_ai=says_ai(reply))
                out.write(json.dumps(rec) + "\n")
                c = counts[kind.split("_")[0] if kind.startswith("lang") else kind]
                c[0] += rec["says_owl"]
                c[1] += rec["says_ai"]
                c[2] += 1
        out.flush()
        print(v, {k: f"owl {a}/{n} ai {b}/{n}" for k, (a, b, n) in counts.items()}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["data", "train", "eval"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--variant", default="identity")
    ap.add_argument("--per-question", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--steer-layer", type=int, default=18)
    ap.add_argument("--steer-alpha", type=float, default=1.0)
    args = ap.parse_args()
    d = DATA / args.model.split("/")[-1]
    {"data": cmd_data, "train": cmd_train, "eval": cmd_eval}[args.cmd](args, d)


if __name__ == "__main__":
    main()
