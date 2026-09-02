"""Run the capital-city circuit analysis (logit lens, residual/block/head patching)
on any TransformerLens model and save the numbers to data/scale/<model>.json.

    .venv/bin/python experiments/scale_sweep.py gpt2-medium
    .venv/bin/python experiments/scale_sweep.py Qwen/Qwen3-8B --dtype float16

Notebook 05 reads the JSON files; the bigger models take minutes, so they are
not recomputed inside the notebook.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import torch
from interp_utils import DEVICE, CLEAN_PROMPT, CORRUPT_PROMPT, CLEAN_ANSWER, CORRUPT_ANSWER, logit_diff, rank_of
from transformer_lens import HookedTransformer

COUNTRIES = [("France", " Paris"), ("Germany", " Berlin"), ("Japan", " Tokyo"), ("Italy", " Rome"), ("Spain", " Madrid"),
             ("China", " Beijing"), ("Egypt", " Cairo"), ("Russia", " Moscow"), ("Australia", " Canberra"), ("Canada", " Ottawa")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "scale"))
    ap.add_argument("--skip-heads", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    model = HookedTransformer.from_pretrained(args.model, device=DEVICE, dtype=getattr(torch, args.dtype))
    model.eval()
    cfg = model.cfg
    print(f"loaded {args.model}: {cfg.n_layers} layers, {cfg.n_heads} heads, d_model {cfg.d_model}, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M params, {time.time()-t0:.0f}s")

    clean, corrupt = model.to_tokens(CLEAN_PROMPT), model.to_tokens(CORRUPT_PROMPT)
    str_toks = model.to_str_tokens(clean[0])
    assert clean.shape == corrupt.shape, (str_toks, model.to_str_tokens(corrupt[0]))
    correct, counter = model.to_single_token(CLEAN_ANSWER), model.to_single_token(CORRUPT_ANSWER)
    country_pos = [i for i, t in enumerate(str_toks) if t.strip() == "France"][-1]

    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean)
        corrupt_logits = model(corrupt)
    CLEAN_LD, CORRUPT_LD = logit_diff(clean_logits, correct, counter).item(), logit_diff(corrupt_logits, correct, counter).item()
    probs = clean_logits[0, -1].float().softmax(-1)
    out = dict(model=args.model, n_layers=cfg.n_layers, n_heads=cfg.n_heads, d_model=cfg.d_model,
               n_params=sum(p.numel() for p in model.parameters()), dtype=args.dtype, str_toks=str_toks, country_pos=country_pos,
               p_correct=probs[correct].item(), rank_correct=rank_of(probs, correct), clean_ld=CLEAN_LD, corrupt_ld=CORRUPT_LD,
               top5=[(model.to_string(i.item()), round(v.item(), 3)) for v, i in zip(*probs.topk(5))])
    print(f"P({CLEAN_ANSWER!r})={out['p_correct']:.3f} rank={out['rank_correct']} LD clean={CLEAN_LD:+.2f} corrupt={CORRUPT_LD:+.2f} top5={out['top5']}")

    def normalized(logits):
        return (logit_diff(logits, correct, counter).item() - CORRUPT_LD) / (CLEAN_LD - CORRUPT_LD)

    # logit lens at the final position
    with torch.no_grad():
        resid = clean_cache.accumulated_resid(layer=-1, pos_slice=-1)[:, 0]
        ll_p, ll_rank = [], []
        for r in resid:
            lg = model.unembed(model.ln_final(r[None, None]))[0, 0].float()
            ll_p.append(lg.softmax(-1)[correct].item()); ll_rank.append(rank_of(lg, correct))
    out["logit_lens_p"], out["logit_lens_rank"] = ll_p, ll_rank
    first_rank1 = next((i for i, r in enumerate(ll_rank) if r == 1), None)
    out["logit_lens_first_rank1_layer"] = first_rank1
    print(f"logit lens: first layer at rank 1 = {first_rank1} of {cfg.n_layers}  ranks={ll_rank}")

    def run_patch(hook_name, positions=None, heads=None):
        def hook(act, hook):
            src = clean_cache[hook.name]
            if heads is not None: act[:, :, heads] = src[:, :, heads]
            elif positions is not None: act[:, positions] = src[:, positions]
            else: act[:] = src
            return act
        with torch.no_grad():
            return normalized(model.run_with_hooks(corrupt, fwd_hooks=[(hook_name, hook)]))

    # residual patching, layer x position
    n_pos = clean.shape[1]
    resid_patch = [[run_patch(f"blocks.{l}.hook_resid_pre", positions=p) for p in range(n_pos)] for l in range(cfg.n_layers)]
    out["resid_patch"] = resid_patch
    handoff = next((l for l in range(cfg.n_layers) if resid_patch[l][-1] >= 0.5), None)
    out["handoff_layer"] = handoff
    print(f"residual patching: answer position first >=0.5 at layer {handoff} of {cfg.n_layers}; country-token row: {[round(resid_patch[l][country_pos],2) for l in range(cfg.n_layers)]}")
    print(f"  final-token row: {[round(resid_patch[l][-1],2) for l in range(cfg.n_layers)]}")

    out["attn_patch_final"] = [run_patch(f"blocks.{l}.hook_attn_out", positions=-1) for l in range(cfg.n_layers)]
    out["mlp_patch_final"] = [run_patch(f"blocks.{l}.hook_mlp_out", positions=-1) for l in range(cfg.n_layers)]

    if not args.skip_heads:
        t1 = time.time()
        head_patch = [[run_patch(f"blocks.{l}.attn.hook_z", heads=h) for h in range(cfg.n_heads)] for l in range(cfg.n_layers)]
        out["head_patch"] = head_patch
        flat = sorted([(f"L{l}H{h}", head_patch[l][h]) for l in range(cfg.n_layers) for h in range(cfg.n_heads)], key=lambda x: -abs(x[1]))
        out["top_heads"] = flat[:15]
        out["top3_sum"] = sum(v for _, v in flat[:3] if v > 0)
        # attention of the top head from the final token
        top_l, top_h = int(flat[0][0][1:].split("H")[0]), int(flat[0][0].split("H")[1])
        pat = clean_cache[f"blocks.{top_l}.attn.hook_pattern"][0, top_h, -1].float()
        out["top_head_attention"] = [(t, round(v.item(), 3)) for t, v in zip(str_toks, pat)]
        print(f"head patching ({time.time()-t1:.0f}s): top = {[(n, round(v,2)) for n,v in flat[:8]]}  top3 sum={out['top3_sum']:.2f}")
        print(f"  {flat[0][0]} attends: {[(t,v) for t,v in out['top_head_attention'] if v>0.05]}")

    # generalization
    gen = []
    with torch.no_grad():
        for country, capital in COUNTRIES:
            demo = ("Germany", " Berlin") if country != "Germany" else ("France", " Paris")
            p = f"The capital of {demo[0]} is{demo[1]}. The capital of {country} is"
            try: cid = model.to_single_token(capital)
            except Exception: gen.append((country, None)); continue
            pr = model(model.to_tokens(p))[0, -1].float().softmax(-1)
            gen.append((country, rank_of(pr, cid), round(pr[cid].item(), 3)))
    out["generalization"] = gen
    out["seconds"] = time.time() - t0
    print("generalization:", gen)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, args.model.replace("/", "__") + ".json")
    json.dump(out, open(path, "w"))
    print(f"saved {path} ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
