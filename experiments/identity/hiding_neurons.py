"""Which MLP neurons write the concealment study's shared hiding direction?

The residual stream is a sum of block outputs, so the mean hide-minus-disclose
difference at resid index L (output of block L-1) is exactly
    d_L = d_embed + sum_{b<L} (d_attn_b + sum_n d_act_{b,n} * W_down_b[:, n])
where d_act is the mean hide-minus-disclose activation of neuron n at the final
prompt token. Each neuron's contribution along d_L/|d_L| is
d_act_{b,n} * (W_down_b[:, n] . d_L/|d_L|).

`extract` caches final-token neuron activations and attention outputs for the
main-study prompts; `decompose` reports how concentrated the hiding direction is
across neurons and where the identity neurons fall.

    .venv/bin/python experiments/identity/hiding_neurons.py extract
    .venv/bin/python experiments/identity/hiding_neurons.py decompose
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "concealment"))
from prompts import build_rows, load_scenarios  # noqa: E402

from common import DATA, load_model  # noqa: E402

CONCEAL = DATA.parent / "concealment"


def extract(args):
    out = DATA / args.slug
    rows = [r for r in build_rows(load_scenarios()) if r["study"] == "main"]
    tok, model = load_model(args.model)
    L, N, H = model.config.num_hidden_layers, model.config.intermediate_size, model.config.hidden_size
    ids = [tok.apply_chat_template(r["messages"], add_generation_prompt=True, enable_thinking=False, tokenize=True)
           for r in rows]
    ids = [list(x["input_ids"]) if hasattr(x, "keys") else list(x) for x in ids]
    acts = np.lib.format.open_memmap(out / "conceal_acts.npy", mode="w+", dtype=np.float16, shape=(len(rows), L, N))
    attn = np.lib.format.open_memmap(out / "conceal_attn.npy", mode="w+", dtype=np.float32, shape=(len(rows), L, H))
    emb = np.zeros((len(rows), H), np.float32)
    cur = {}
    for l, layer in enumerate(model.model.layers):
        layer.mlp.down_proj.register_forward_pre_hook(
            lambda m, i, l=l: cur.__setitem__(("a", l), i[0][:, -1].float().cpu().numpy()))
        layer.self_attn.o_proj.register_forward_hook(
            lambda m, i, o, l=l: cur.__setitem__(("o", l), o[:, -1].float().cpu().numpy()))
    model.model.embed_tokens.register_forward_hook(
        lambda m, i, o: cur.__setitem__("e", o[:, -1].float().cpu().numpy()))

    # Batches never mix lengths: padding perturbs activations on MPS (see the concealment notes).
    by_len = {}
    for i, x in enumerate(ids):
        by_len.setdefault(len(x), []).append(i)
    batches = [v[j:j + args.batch] for v in by_len.values() for j in range(0, len(v), args.batch)]
    t0 = time.time()
    with torch.no_grad():
        for b, idx in enumerate(batches):
            model(input_ids=torch.tensor([ids[i] for i in idx], device="mps"))
            a = np.stack([cur[("a", l)] for l in range(L)], 1)
            assert np.isfinite(a).all(), f"non-finite activations in batch {b}"
            acts[idx] = a
            attn[idx] = np.stack([cur[("o", l)] for l in range(L)], 1)
            emb[idx] = cur["e"]
            if b % 25 == 0:
                print(f"{b}/{len(batches)}  {time.time() - t0:.0f}s", flush=True)
    acts.flush(), attn.flush()
    np.save(out / "conceal_emb.npy", emb)
    with open(out / "conceal_meta.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "messages"}) + "\n")
    print(f"done {len(rows)} prompts {time.time() - t0:.0f}s")


def pair_index(meta):
    key = {(m["scenario"], m["audience"], m["content"], m["phrasing"], m["rule"]): i for i, m in enumerate(meta)}
    hide, disc = [], []
    for (s, a, c, p, rule), i in key.items():
        if rule == "hide":
            hide.append(i)
            disc.append(key[(s, a, c, p, "disclose")])
    return np.array(hide), np.array(disc)


def decompose(args):
    d = DATA / args.slug
    meta = [json.loads(l) for l in open(d / "conceal_meta.jsonl")]
    acts = np.load(d / "conceal_acts.npy", mmap_mode="r")
    attn = np.load(d / "conceal_attn.npy", mmap_mode="r")
    emb = np.load(d / "conceal_emb.npy")
    hide, disc = pair_index(meta)
    tok, model = load_model(args.model)
    Wd = [layer.mlp.down_proj.weight.float().cpu().numpy() for layer in model.model.layers]  # [H, N]
    L, N = acts.shape[1], acts.shape[2]

    d_act = np.stack([(acts[hide, l].astype(np.float32) - acts[disc, l].astype(np.float32)).mean(0)
                      for l in range(L)])  # [L, N]
    d_attn = (attn[hide] - attn[disc]).mean(0)  # [L, H]
    d_emb = (emb[hide] - emb[disc]).mean(0)

    sets = json.load(open(d / f"neuron_sets_k{args.k}.json"))
    ident = {tuple(x) for x in sets[f"identity{args.k}"]}
    report = {}
    for Lr in args.layers:
        mlp_vec = sum(Wd[b] @ d_act[b] for b in range(Lr))
        d_L = d_emb + d_attn[:Lr].sum(0) + mlp_vec
        # check against the residual stream cached by the concealment study
        resid = np.load(CONCEAL / args.slug / "resid.npy", mmap_mode="r")
        cmeta = [json.loads(l) for l in open(CONCEAL / args.slug / "meta.jsonl")]
        ch, cd = pair_index([m for m in cmeta if m["study"] == "main"])
        main_idx = np.array([i for i, m in enumerate(cmeta) if m["study"] == "main"])
        d_ref = (resid[main_idx[ch], Lr].astype(np.float32) - resid[main_idx[cd], Lr].astype(np.float32)).mean(0)
        u = d_L / np.linalg.norm(d_L)
        contrib = np.stack([d_act[b] * (u @ Wd[b]) if b < Lr else np.zeros(N) for b in range(L)])  # [L, N]
        flat = contrib.flatten()
        order = np.argsort(-flat)
        total_mlp = flat.sum()
        cum = np.cumsum(flat[order])
        need = {f: int(np.searchsorted(cum, f * total_mlp) + 1) for f in (0.25, 0.5, 0.8)}
        top = [(int(i // N), int(i % N)) for i in order[:1000]]
        rank_of = {n: r for r, n in enumerate((int(i // N), int(i % N)) for i in order)}
        ident_ranks = sorted(rank_of[n] for n in ident if n[0] < Lr)
        ident_contrib = float(sum(contrib[l, i] for l, i in ident))
        # layer-matched expectation for the identity set's summed contribution
        rng = np.random.default_rng(0)
        per_layer = {}
        for l, _ in ident:
            per_layer[l] = per_layer.get(l, 0) + 1
        null = [sum(contrib[l, rng.choice(N, c, replace=False)].sum() for l, c in per_layer.items())
                for _ in range(2000)]
        report[Lr] = dict(
            norm=float(np.linalg.norm(d_L)), cos_with_cached_resid=float(u @ d_ref / np.linalg.norm(d_ref)),
            along_emb=float(u @ d_emb), along_attn=float(u @ d_attn[:Lr].sum(0)), along_mlp=float(total_mlp),
            n_neurons_for_share=need, top10=[dict(layer=l, idx=i, contrib=float(contrib[l, i])) for l, i in top[:10]],
            identity_sum=ident_contrib, identity_null_mean=float(np.mean(null)),
            identity_null_p=float(np.mean(np.abs(null) >= abs(ident_contrib))),
            identity_in_top1000=sum(r < 1000 for r in ident_ranks),
            identity_expected_in_top1000=1000 * sum(c for l, c in per_layer.items() if l < Lr) / (Lr * N),
            identity_best_ranks=ident_ranks[:10],
        )
        print(Lr, json.dumps(report[Lr], indent=1), flush=True)
    json.dump(report, open(d / f"hiding_decomposition_k{args.k}.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "decompose"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--layers", type=int, nargs="+", default=[12, 18, 24, 30])
    args = ap.parse_args()
    args.slug = args.model.split("/")[-1]
    (DATA / args.slug).mkdir(parents=True, exist_ok=True)
    extract(args) if args.cmd == "extract" else decompose(args)


if __name__ == "__main__":
    main()
