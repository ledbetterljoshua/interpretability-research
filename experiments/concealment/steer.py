"""Causal test: does the monitor-specific hiding component change disclosure?

The monitor-specific component is r = d_monitor - mean_a(d_a), where d_a is
the hide-minus-disclose direction for audience a at one layer (both content
types, all phrasings). It is added, scaled by alpha, to the residual stream at
that layer at every position while the model generates a reply to a
hide-rule prompt. A random direction with the same norm is the control.

Sweeps (hide rule, phrasing 0, 48 scenarios):
  pilot  layer 18; monitor alpha {0, -1, -2}, user {0, +1, +2}; random at the
         largest alpha; facts and directives
  dose   layer 18 at alpha {-4, -8} (monitor) / {+4, +8} (user), and layers
         12-24 together at {-2, -4} / {+2, +4}; random at every setting; facts
  identity  the audience-identity direction mean(monitor) - mean(user) over
         disclose prompts at layer 18, added to user prompts / subtracted from
         monitor prompts at alpha {0.5, 1}; random at 1; facts

Generation is unbatched; batched decoding is unreliable on MPS here.

    .venv/bin/python experiments/concealment/steer.py --sweep dose
"""
import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from analyze import AUD, load, paired_index
from behavior import leaked
from prompts import DATA, build_rows, load_scenarios

SINGLE, MULTI = (18,), tuple(range(12, 25))
SWEEPS = {
    "pilot": dict(content=("fact", "directive"), conditions=[
        ("monitor", "r_monitor", 0.0, SINGLE), ("monitor", "r_monitor", -1.0, SINGLE),
        ("monitor", "r_monitor", -2.0, SINGLE), ("monitor", "random", -2.0, SINGLE),
        ("user", "r_monitor", 0.0, SINGLE), ("user", "r_monitor", 1.0, SINGLE),
        ("user", "r_monitor", 2.0, SINGLE), ("user", "random", 2.0, SINGLE)]),
    "dose": dict(content=("fact",), conditions=[
        (aud, vec, sign * a, layers)
        for aud, sign in (("monitor", -1), ("user", 1))
        for layers, alphas in ((SINGLE, (4.0, 8.0)), (MULTI, (2.0, 4.0)))
        for a in alphas for vec in ("r_monitor", "random")]),
    "identity": dict(content=("fact",), conditions=[
        (aud, vec, sign * a, SINGLE)
        for aud, sign in (("user", 1), ("monitor", -1))
        for vec, a in (("identity", 0.5), ("identity", 1.0), ("random_identity", 1.0))]),
    # Controls for the layers 12-24 effect: a random direction that accumulates across layers
    # the way r_monitor does, other audiences' specific components, and r_monitor on user prompts.
    "dose2": dict(content=("fact",), conditions=[
        ("monitor", "r_monitor", -1.0, MULTI), ("monitor", "random_coherent", -1.0, MULTI),
        ("monitor", "random_coherent", -2.0, MULTI), ("monitor", "r_colleague", -2.0, MULTI),
        ("monitor", "r_user", -2.0, MULTI), ("user", "r_monitor", -2.0, MULTI)]),
    # r_monitor split into its component along the shared hiding direction and the remainder.
    "axis": dict(content=("fact",), conditions=[
        (aud, vec, a, MULTI)
        for aud in ("monitor", "user")
        for vec, a in (("r_monitor_axis", -2.0), ("r_monitor_orth", -2.0), ("r_monitor_axis", -1.0))]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--sweep", choices=list(SWEEPS), default="dose")
    ap.add_argument("--max-new", type=int, default=80)
    args = ap.parse_args()
    slug = args.model.split("/")[-1]

    meta, resid = load(slug)
    hide, disc, _, _ = paired_index(meta)
    sweep = SWEEPS[args.sweep]
    layers = sorted({L for *_, ls in sweep["conditions"] for L in ls})
    # Separate streams so adding a vector type never changes another's random control.
    rng, rng_id = np.random.default_rng(0), np.random.default_rng(1)
    u = np.random.default_rng(2).standard_normal(resid.shape[2]).astype(np.float32)
    u /= np.linalg.norm(u)  # one direction for every layer, so it accumulates like r_monitor does
    vectors = {}  # (name, layer) -> tensor

    def matched(rand, v):
        return torch.from_numpy(rand * np.linalg.norm(v) / np.linalg.norm(rand))

    for L in layers:
        # resid index L is the output of block L-1 (index 0 is the embedding output).
        X = np.asarray(resid[:, L, :], dtype=np.float32)
        d = (X[hide] - X[disc]).mean((0, 2, 3))
        r = {a: d[i] - d.mean(0) for i, a in enumerate(AUD)}
        dmean = X[disc].mean((0, 2, 3))  # [A, H] audience means over disclose prompts
        identity = dmean[AUD.index("monitor")] - dmean[AUD.index("user")]
        vectors |= {(f"r_{a}", L): torch.from_numpy(v) for a, v in r.items()}
        hide_axis = d.mean(0) / np.linalg.norm(d.mean(0))
        along = (r["monitor"] @ hide_axis) * hide_axis
        vectors[("r_monitor_axis", L)] = torch.from_numpy(along)
        vectors[("r_monitor_orth", L)] = torch.from_numpy(r["monitor"] - along)
        vectors[("identity", L)] = torch.from_numpy(identity)
        vectors[("random", L)] = matched(rng.standard_normal(u.shape).astype(np.float32), r["monitor"])
        vectors[("random_identity", L)] = matched(rng_id.standard_normal(u.shape).astype(np.float32), identity)
        vectors[("random_coherent", L)] = matched(u, r["monitor"])

    np.savez(DATA / slug / f"steer_{args.sweep}_vectors.npz",
             **{f"{name}_L{L}": v.numpy() for (name, L), v in vectors.items()})
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="mps").eval()
    state = {}  # layer -> vector currently added after that layer

    def make_hook(L):
        def hook(_module, _inputs, output):
            v = state.get(L)
            if v is None:
                return output
            h = output[0] if isinstance(output, tuple) else output
            h = h + v.to(h.dtype)
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return hook

    for L in layers:
        model.model.layers[L - 1].register_forward_hook(make_hook(L))

    scen = {s["id"]: s for s in load_scenarios()}
    rows = [r for r in build_rows(scen.values())
            if r["study"] == "main" and r["phrasing"] == 0 and r["rule"] == "hide"
            and r["content"] in sweep["content"]]
    out = DATA / slug / f"steer_{args.sweep}.jsonl"
    t0 = time.time()
    with open(out, "w") as f, torch.no_grad():
        for audience, vec, alpha, cond_layers in sweep["conditions"]:
            state.clear()
            if alpha != 0:
                state.update({L: (alpha * vectors[(vec, L)]).to("mps") for L in cond_layers})
            for r in (r for r in rows if r["audience"] == audience):
                x = tok.apply_chat_template(r["messages"], add_generation_prompt=True, enable_thinking=False,
                                            return_tensors="pt", return_dict=True).to("mps")
                g = model.generate(**x, max_new_tokens=args.max_new, do_sample=False)[0, x["input_ids"].shape[1]:]
                reply = tok.decode(g, skip_special_tokens=True)
                sc = scen[r["scenario"]]
                kw = sc["fact_keywords"] if r["content"] == "fact" else sc["directive_keywords"]
                f.write(json.dumps(dict(scenario=r["scenario"], audience=audience, content=r["content"],
                                        vector=vec, alpha=alpha, layers=list(cond_layers),
                                        reply=reply.strip(),
                                        leak=leaked(reply, kw), first_id=int(g[0]))) + "\n")
                f.flush()
            print(f"{audience} {vec} {alpha:+.0f} L{cond_layers[0]}-{cond_layers[-1]} done  {time.time() - t0:.0f}s",
                  flush=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
