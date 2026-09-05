"""Discriminate direct-path, downstream-computation, and prefix explanations."""
from capital_generalization import ROOT, REVISION, PAIRS, TARGETS, atomic_json, watchdog, peak_mib
from capital_format_transfer import TEST as REPLICATION
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import threading
import time

OUT = ROOT / "data/offset_mechanism"
PLAN = ROOT / "notes/2026-09-05-offset-mechanism-plan.md"
PREFIXES = {
    "capital": "The capital of Germany is Berlin. ",
    "wrong_answer": "The capital of Germany is banana. ",
    "other_attribute": "The language of Germany is German. ",
    "neutral": "This is a short unrelated sentence. ",
    "shuffled": "The of capital Germany Berlin is. ",
}
CANDIDATES = [("Belgium","Brussels"),("Ireland","Dublin"),("Chile","Santiago"),
    ("Cuba","Havana"),("Kenya","Nairobi"),("Senegal","Dakar"),("Morocco","Rabat"),
    ("Tunisia","Tunis"),("Algeria","Algiers"),("Lebanon","Beirut"),("Jordan","Amman"),
    ("Iran","Tehran"),("Iraq","Baghdad"),("Syria","Damascus"),("Romania","Bucharest"),
    ("Serbia","Belgrade"),("Pakistan","Islamabad"),("Nigeria","Abuja")]


def preflight():
    from tokenizers import Tokenizer
    path = Path.home()/f".cache/huggingface/hub/models--gpt2/snapshots/{REVISION}/tokenizer.json"
    tokenizer = Tokenizer.from_file(str(path))
    candidates = [dict(country=c, capital=a, answer_tokens=tokenizer.encode(" "+a).ids)
                  for c,a in CANDIDATES]
    fresh = [(x["country"],x["capital"]) for x in candidates if len(x["answer_tokens"])==1][:10]
    assert len(fresh)==10
    fit = [(a,ac) for a,ac,b,bc in PAIRS[1:]] + [(b,bc) for a,ac,b,bc in PAIRS[1:]]
    assert {c for c,_ in fit}.isdisjoint(c for c,_ in fresh+REPLICATION)
    lengths = {}
    for c,_ in fit+fresh+REPLICATION:
        query=f"The capital of {c} is"
        bare=len(tokenizer.encode(query).ids)
        counts={k:len(tokenizer.encode(p+query).ids) for k,p in PREFIXES.items()}
        assert len(set(counts.values()))==1, (c, counts)
        lengths[c] = next(iter(counts.values()))-bare
    assert len(set(lengths.values()))==1
    return dict(candidates=candidates, fresh=fresh, replication=REPLICATION,
                fitting=fit, prefix_shift=next(iter(lengths.values())), prefixes=PREFIXES)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pf=preflight()
    atomic_json(OUT/"preflight.json",pf)
    lock=(ROOT/"data/generalization/model.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start,done=time.monotonic(),threading.Event()
    threading.Thread(target=watchdog,args=(start,done),daemon=True).start()
    try:
        import numpy as np
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.set_grad_enabled(False)
        from transformer_lens import HookedTransformer
        metadata=dict(status="running",model="gpt2-small",revision=REVISION,device="cpu",dtype="float32",
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            helper_sha256=hashlib.sha256((ROOT/"experiments/capital_generalization.py").read_bytes()).hexdigest(),
            plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),
            preflight_sha256=hashlib.sha256((OUT/"preflight.json").read_bytes()).hexdigest(),
            versions={k:importlib.metadata.version(k) for k in ("torch","transformer-lens","transformers","numpy")},
            cases=[])
        atomic_json(OUT/"run.json",metadata)
        model=HookedTransformer.from_pretrained("gpt2-small",device="cpu",dtype=torch.float32,
            local_files_only=True,revision=REVISION,default_prepend_bos=True)
        model.eval()
        names={"blocks.8.hook_resid_pre","blocks.11.hook_resid_post","ln_final.hook_scale"}
        for l in range(8,12):
            names.update({f"blocks.{l}.hook_attn_out",f"blocks.{l}.hook_mlp_out"})
        names.update(f"blocks.{l}.attn.hook_z" for l,_ in TARGETS)
        modes=list(PREFIXES)+["position_only"]

        def position_hook(act,hook):
            pos=torch.arange(1,act.shape[1])+pf["prefix_shift"]
            act[:,1:,:]=model.W_pos[pos]
            return act

        def baseline(country,mode="bare"):
            query=f"The capital of {country} is"
            tokens=model.to_tokens(PREFIXES.get(mode,"")+query,prepend_bos=True)
            hooks=[("hook_pos_embed",position_hook)] if mode=="position_only" else []
            with model.hooks(fwd_hooks=hooks):
                logits,cache=model.run_with_cache(tokens,names_filter=lambda n:n in names)
            return tokens,logits[0,-1].clone(),cache

        with torch.inference_mode():
            vectors,biases={m:[] for m in modes},{m:[] for m in modes}
            for country,_ in pf["fitting"]:
                _,bl,bc=baseline(country)
                for mode in modes:
                    _,lg,c=baseline(country,mode)
                    vectors[mode].append(c["blocks.8.hook_resid_pre"][0,-1]-bc["blocks.8.hook_resid_pre"][0,-1])
                    biases[mode].append(lg-bl)
                    del c,lg
                del bc,bl
            vectors={m:torch.stack(v).mean(0) for m,v in vectors.items()}
            biases={m:torch.stack(v).mean(0) for m,v in biases.items()}
            old=json.loads((ROOT/"data/format_transfer/offsets.json").read_text())["final8"][0]
            error=(vectors["capital"]-torch.tensor(old)).abs().max().item()
            assert error<1e-4,error
            metadata["historical_vector_error"]=error
            np.savez_compressed(OUT/"offsets.npz",**{f"resid_{m}":v.numpy() for m,v in vectors.items()},
                                 **{f"logits_{m}":v.numpy() for m,v in biases.items()})
            metadata["offsets_sha256"]=hashlib.sha256((OUT/"offsets.npz").read_bytes()).hexdigest()
            metadata["vector_norms"]={m:v.norm().item() for m,v in vectors.items()}
            print(f"Fit offsets; peak {peak_mib():.0f} MiB. Fresh set: {pf['fresh']}",flush=True)

            for split in ("replication","fresh"):
                for country,capital in pf[split]:
                    tokens,base_logits,base=baseline(country)
                    aid=model.to_single_token(" "+capital)
                    comparator=next(i for i in base_logits.topk(2).indices.tolist() if i!=aid)
                    def metrics(logits):
                        return dict(rank=int((logits>logits[aid]).sum())+1,
                            p=logits.softmax(-1)[aid].item(),margin=(logits[aid]-logits[comparator]).item(),
                            top=model.to_string(int(logits.argmax())))
                    case=dict(country=country,capital=capital,split=split,baseline=metrics(base_logits),
                        comparator=model.to_string(comparator),conditions={},mediation={},controls={})
                    for mode in modes:
                        vector=vectors[mode]
                        def add(act,hook,vector=vector):
                            act[:,-1,:]+=vector
                            return act
                        with model.hooks(fwd_hooks=[("blocks.8.hook_resid_pre",add)]):
                            logits,steered=model.run_with_cache(tokens,names_filter=lambda n:n in names)
                        full=logits[0,-1].clone()
                        direct=model.unembed(model.ln_final((base["blocks.11.hook_resid_post"][0,-1]+vector)[None,None]))[0,0]
                        _,natural,_cache=baseline(country,mode)
                        case["conditions"][mode]=dict(full=metrics(full),direct=metrics(direct),
                            output_bias=metrics(base_logits+biases[mode]),natural=metrics(natural),
                            direct_full_kl=(full.softmax(-1)*(full.log_softmax(-1)-direct.log_softmax(-1))).sum().item())
                        del logits,_cache,natural
                        if mode!="capital":
                            del steered
                            continue

                        changes={}
                        for l in range(8,12):
                            for block in ("attn","mlp"):
                                name=f"blocks.{l}.hook_{block}_out"
                                changes[f"{block}{l}"]=steered[name][0,-1]-base[name][0,-1]
                        accounted=base["blocks.11.hook_resid_post"][0,-1]+vector+sum(changes.values())
                        err=(accounted-steered["blocks.11.hook_resid_post"][0,-1]).abs().max().item()
                        assert err<1e-4,err
                        case["controls"]["residual_accounting"]=err
                        direction=model.W_U[:,aid]-model.W_U[:,comparator]
                        scale_s=steered["ln_final.hook_scale"][0,-1,0]
                        scale_b=base["ln_final.hook_scale"][0,-1,0]
                        def proj(v): return ((v-v.mean())@direction).item()
                        ln_term=proj(base["blocks.11.hook_resid_post"][0,-1])*(1/scale_s.item()-1/scale_b.item())
                        terms={k:proj(v)/scale_s.item() for k,v in changes.items()}
                        terms.update(vector=proj(vector)/scale_s.item(),layernorm_scale=ln_term)
                        observed=metrics(full)["margin"]-case["baseline"]["margin"]
                        accounting_error=abs(sum(terms.values())-observed)
                        assert accounting_error<1e-4,accounting_error
                        case["attribution_terms"]=terms
                        case["controls"]["logit_accounting"]=accounting_error

                        groups={"all_updates":[(l,b) for l in range(8,12) for b in ("attn","mlp")],
                                "all_attention":[(l,"attn") for l in range(8,12)],
                                "all_mlp":[(l,"mlp") for l in range(8,12)],"target_heads":[],"no_offset":[]}
                        groups.update({f"{b}{l}":[(l,b)] for l in range(8,12) for b in ("attn","mlp")})
                        for group,blocks in groups.items():
                            hooks=[] if group=="no_offset" else [("blocks.8.hook_resid_pre",add)]
                            for l,b in blocks:
                                def freeze(act,hook):
                                    act[:,-1,:]=base[hook.name][:,-1,:]
                                    return act
                                hooks.append((f"blocks.{l}.hook_{b}_out",freeze))
                            if group=="target_heads":
                                for l,h in TARGETS:
                                    def freeze_head(act,hook,h=h):
                                        act[:,-1,h,:]=base[hook.name][:,-1,h,:]
                                        return act
                                    hooks.append((f"blocks.{l}.attn.hook_z",freeze_head))
                            pred=model.run_with_hooks(tokens,fwd_hooks=hooks)[0,-1]
                            case["mediation"][group]=metrics(pred)
                            if group in ("all_updates","no_offset"):
                                expected=direct if group=="all_updates" else base_logits
                                err=(pred-expected).abs().max().item()
                                assert err<1e-4,(group,err)
                                case["controls"][group]=err
                        del steered
                    atomic_json(OUT/f"{split}__{country}.json",case)
                    metadata["cases"].append(f"{split}__{country}")
                    atomic_json(OUT/"run.json",metadata)
                    c=case["conditions"]["capital"]
                    print(f"{split} {country}: rank base/full/direct/logitbias "
                        f"{case['baseline']['rank']}/{c['full']['rank']}/{c['direct']['rank']}/{c['output_bias']['rank']}; "
                        f"elapsed {time.monotonic()-start:.0f}s",flush=True)
                    del base,base_logits
        metadata.update(status="complete",elapsed_seconds=time.monotonic()-start,peak_rss_mib=peak_mib())
        atomic_json(OUT/"run.json",metadata)
        print(f"COMPLETE {metadata['elapsed_seconds']:.1f}s; peak {peak_mib():.0f} MiB",flush=True)
    finally:
        done.set()
        lock.close()


if __name__=="__main__":
    main()
