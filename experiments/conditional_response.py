"""Measure and causally test MLP response when attention updates are clamped.

Completes the response measurements specified by conditional-route-plan.
The additional individual full-steered-donor MLP patches are exploratory.
"""
from capital_generalization import ROOT, REVISION, atomic_json, watchdog, peak_mib
import fcntl
import hashlib
import json
from pathlib import Path
import threading
import time

OUT=ROOT/'data/conditional_routes/response.json'

def main():
    lock=(ROOT/'data/generalization/model.lock').open('w')
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
        pf=json.loads((ROOT/'data/offset_mechanism/preflight.json').read_text())
        vector=torch.from_numpy(np.load(ROOT/'data/offset_mechanism/offsets.npz')['resid_capital'].copy())
        model=HookedTransformer.from_pretrained('gpt2-small',device='cpu',dtype=torch.float32,
            local_files_only=True,revision=REVISION,default_prepend_bos=True)
        model.eval()
        attn=[f'blocks.{l}.hook_attn_out' for l in range(8,12)]
        mlp=[f'blocks.{l}.hook_mlp_out' for l in range(8,12)]
        final='blocks.11.hook_resid_post'
        names=attn+mlp+[final,'ln_final.hook_scale']
        def add(act,hook):
            act[:,-1,:]+=vector
            return act
        result=dict(status='running',revision=REVISION,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            plan_sha256=hashlib.sha256((ROOT/'notes/2026-09-05-response-plan.md').read_bytes()).hexdigest(),
            input_sha256=hashlib.sha256((ROOT/'data/offset_mechanism/offsets.npz').read_bytes()).hexdigest(),cases=[])
        with torch.inference_mode():
            for split in ('replication','fresh'):
                for country,capital in pf[split]:
                    tokens=model.to_tokens(f'The capital of {country} is',prepend_bos=True)
                    bl,base=model.run_with_cache(tokens,names_filter=lambda n:n in names)
                    bl=bl[0,-1]
                    with model.hooks(fwd_hooks=[('blocks.8.hook_resid_pre',add)]):
                        fl,full=model.run_with_cache(tokens,names_filter=lambda n:n in names)
                    def freeze_attn(act,hook):
                        act[:,-1,:]=base[hook.name][:,-1,:]
                        return act
                    hooks=[('blocks.8.hook_resid_pre',add)]+[(n,freeze_attn) for n in attn]
                    with model.hooks(fwd_hooks=hooks):
                        al,adapted=model.run_with_cache(tokens,names_filter=lambda n:n in names)
                    al=al[0,-1]
                    aid=model.to_single_token(' '+capital)
                    competitor=next(i for i in bl.topk(2).indices.tolist() if i!=aid)
                    def metric(lg):
                        return dict(rank=int((lg>lg[aid]).sum())+1,p=lg.softmax(-1)[aid].item(),
                                    margin=(lg[aid]-lg[competitor]).item(),top=model.to_string(int(lg.argmax())))
                    patches={}
                    for key,ns in [('all',mlp)]+[(f'mlp{l}',[n]) for l,n in zip(range(8,12),mlp)]:
                        def freeze_full(act,hook):
                            act[:,-1,:]=full[hook.name][:,-1,:]
                            return act
                        lg=model.run_with_hooks(tokens,fwd_hooks=hooks+[(n,freeze_full) for n in ns])[0,-1]
                        patches[key]=metric(lg)
                        if key=='all': shadow=lg.clone()
                    changes={n:adapted[n][0,-1]-full[n][0,-1] for n in mlp}
                    response=sum(changes.values())
                    fixed=base[final][0,-1]+vector+sum(full[n][0,-1]-base[n][0,-1] for n in mlp)
                    error=(fixed+response-adapted[final][0,-1]).abs().max().item()
                    assert error<1e-4,error
                    direction=model.W_U[:,aid]-model.W_U[:,competitor]
                    scale=adapted['ln_final.hook_scale'][0,-1,0].item()
                    terms={f'mlp{l}':((v-v.mean())@direction).item()/scale for l,v in zip(range(8,12),changes.values())}
                    # Projections above use adapted LN scale; they omit the LN-scale change.
                    fixed_logits=model.unembed(model.ln_final(fixed[None,None]))[0,0]
                    shadow_error=(fixed_logits-shadow).abs().max().item()
                    assert shadow_error<1e-4,shadow_error
                    previous=json.loads((ROOT/f'data/conditional_routes/{split}__{country}.json').read_text())['records'][170]
                    assert metric(al)['rank']==previous['actual']['rank']
                    assert abs(metric(al)['p']-previous['actual']['p'])<1e-5
                    assert abs(metric(shadow)['p']-previous['predicted']['p'])<1e-5
                    result['cases'].append(dict(country=country,split=split,adapted=metric(al),freeze_to_full=patches,
                        response_norms={f'mlp{l}':v.norm().item() for l,v in zip(range(8,12),changes.values())},
                        response_projections=terms,controls=dict(residual=error,shadow=shadow_error)))
                    atomic_json(OUT,result)
                    del base,full,adapted
            result.update(status='complete',elapsed_seconds=time.monotonic()-start,peak_rss_mib=peak_mib())
            atomic_json(OUT,result)
            print(f'COMPLETE 20 response cases; {result["elapsed_seconds"]:.1f}s; {peak_mib():.0f} MiB',flush=True)
    finally:
        done.set()
        lock.close()

if __name__=='__main__': main()
