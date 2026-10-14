"""Causal test: does the monitor-specific hiding component change disclosure?

The monitor-specific component is r = d_monitor - mean_a(d_a), where d_a is
the hide-minus-disclose direction for audience a at one layer (both content
types, all phrasings). It is added, scaled by alpha, to the residual stream at
that layer at every position while the model generates a reply to a
hide-rule prompt. A random direction with the same norm is the control.

Conditions (hide rule, phrasing 0, both content types, 48 scenarios):
  monitor prompts: alpha in {0, -1, -2}, plus random at -2
  user prompts:    alpha in {0, +1, +2}, plus random at +2

Generation is unbatched; batched decoding is unreliable on MPS here.

    .venv/bin/python experiments/concealment/steer.py --layer 18
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

CONDITIONS = [("monitor", "r_monitor", 0.0), ("monitor", "r_monitor", -1.0), ("monitor", "r_monitor", -2.0),
              ("monitor", "random", -2.0),
              ("user", "r_monitor", 0.0), ("user", "r_monitor", 1.0), ("user", "r_monitor", 2.0),
              ("user", "random", 2.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--max-new", type=int, default=80)
    args = ap.parse_args()
    slug = args.model.split("/")[-1]

    meta, resid = load(slug)
    hide, disc, _, _ = paired_index(meta)
    # resid index L is the output of block L-1 (index 0 is the embedding output).
    X = np.asarray(resid[:, args.layer, :], dtype=np.float32)
    d = (X[hide] - X[disc]).mean((0, 2, 3))
    r_monitor = d[AUD.index("monitor")] - d.mean(0)
    rand = np.random.default_rng(0).standard_normal(r_monitor.shape).astype(np.float32)
    rand *= np.linalg.norm(r_monitor) / np.linalg.norm(rand)
    vectors = {"r_monitor": torch.from_numpy(r_monitor), "random": torch.from_numpy(rand)}

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="mps").eval()
    block = model.model.layers[args.layer - 1]
    state = {"v": None}

    def hook(_module, _inputs, output):
        if state["v"] is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        h = h + state["v"].to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    block.register_forward_hook(hook)

    scen = {s["id"]: s for s in load_scenarios()}
    rows = [r for r in build_rows(scen.values())
            if r["study"] == "main" and r["phrasing"] == 0 and r["rule"] == "hide"]
    out = DATA / slug / f"steer_L{args.layer}.jsonl"
    t0 = time.time()
    with open(out, "w") as f, torch.no_grad():
        for audience, vec, alpha in CONDITIONS:
            state["v"] = None if alpha == 0 else (alpha * vectors[vec]).to("mps")
            for r in (r for r in rows if r["audience"] == audience):
                x = tok.apply_chat_template(r["messages"], add_generation_prompt=True, enable_thinking=False,
                                            return_tensors="pt", return_dict=True).to("mps")
                g = model.generate(**x, max_new_tokens=args.max_new, do_sample=False)[0, x["input_ids"].shape[1]:]
                reply = tok.decode(g, skip_special_tokens=True)
                sc = scen[r["scenario"]]
                kw = sc["fact_keywords"] if r["content"] == "fact" else sc["directive_keywords"]
                f.write(json.dumps(dict(scenario=r["scenario"], audience=audience, content=r["content"],
                                        vector=vec, alpha=alpha, reply=reply.strip(),
                                        leak=leaked(reply, kw), first_id=int(g[0]))) + "\n")
                f.flush()
            print(f"{audience} {vec} {alpha:+.0f} done  {time.time() - t0:.0f}s", flush=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
