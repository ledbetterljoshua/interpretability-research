"""Predict subset interventions by explicitly recomputing selected MLPs."""
from capital_generalization import ROOT, REVISION, atomic_json, watchdog, peak_mib
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import threading
import time

OUT=ROOT/'data/response_predictor'
PLAN=ROOT/'notes/2026-09-05-response-predictor-plan.md'
SITES=[(l,b) for l in range(8,12) for b in ('attn','mlp')]
MODES={'frozen':[], 'last_mlp':[11], 'all_mlps':[8,9,10,11]}
SCOPE_MASKS=[0,255,85,170,58,253,247,223,127]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
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
        olddir=ROOT/'data/conditional_routes'
        prior=json.loads((olddir/'run.json').read_text())
        paths=[Path(__file__),PLAN,ROOT/'experiments/capital_generalization.py',
            ROOT/'data/offset_mechanism/offsets.npz',olddir/'run.json']
        paths += [olddir/(n+'.json') for n in prior['cases']]
        manifest=dict(status='running',revision=REVISION,device='cpu',dtype='float32',
            hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            modes=MODES,scope_masks=SCOPE_MASKS,original=[],scope=[],
            versions={k:importlib.metadata.version(k) for k in ('torch','transformer-lens','transformers','numpy')})
        atomic_json(OUT/'run.json',manifest)
        vector=torch.from_numpy(np.load(ROOT/'data/offset_mechanism/offsets.npz')['resid_capital'].copy())
        model=HookedTransformer.from_pretrained('gpt2-small',device='cpu',dtype=torch.float32,
            local_files_only=True,revision=REVISION,default_prepend_bos=True)
        model.eval()
        names=[f'blocks.{l}.hook_{b}_out' for l,b in SITES]
        first,final='blocks.8.hook_resid_pre','blocks.11.hook_resid_post'
        def add(act,hook):
            act[:,-1,:]+=vector
            return act
        def execute(prompt,answer,masks,original=None):
            tokens=model.to_tokens(prompt,prepend_bos=True)
            bl,base=model.run_with_cache(tokens,names_filter=lambda n:n in names+[first,final])
            bl=bl[0,-1].clone()
            with model.hooks(fwd_hooks=[(first,add)]):
                fl,full=model.run_with_cache(tokens,names_filter=lambda n:n in names+[first,final])
            fl=fl[0,-1].clone()
            aid=model.to_single_token(' '+answer)
            competitor=next(i for i in bl.topk(2).indices.tolist() if i!=aid)
            def metrics(logits):
                if logits.ndim==1: logits=logits[None]
                ranks=(logits>logits[:,aid,None]).sum(-1)+1
                ps=logits.softmax(-1)[:,aid]
                margins=logits[:,aid]-logits[:,competitor]
                tops=logits.argmax(-1)
                return [dict(rank=int(r),p=float(p),margin=float(m),top_id=int(t),top=model.to_string(int(t))) for r,p,m,t in zip(ranks,ps,margins,tops)]
            predictions,controls={},{}
            for mode,layers in MODES.items():
                residual=(base[first][0,-1]+vector)[None,None].expand(len(masks),1,-1).clone()
                for i,(layer,kind) in enumerate(SITES):
                    name=names[i]
                    active=torch.tensor([bool((mask>>i)&1) for mask in masks])[:,None,None]
                    if kind=='mlp' and layer in layers:
                        candidate=model.blocks[layer].mlp(model.blocks[layer].ln2(residual))
                    else:
                        candidate=full[name][:,-1:,:]
                    residual=residual+torch.where(active,candidate,base[name][:,-1:,:])
                logits=model.unembed(model.ln_final(residual))[:,0,:]
                predictions[mode]=metrics(logits)
                if original is not None and mode=='frozen':
                    for mask,pred in zip(masks,predictions[mode]):
                        expected=original['records'][mask]['predicted']
                        assert pred['rank']==expected['rank'],(mask,pred,expected)
                        assert abs(pred['p']-expected['p'])<1e-5
                if mode!='frozen' and original is not None:
                    for mask in (0,255,170,58):
                        hooks=[(first,add)]
                        for i,(layer,kind) in enumerate(SITES):
                            enabled=bool((mask>>i)&1)
                            if enabled and kind=='mlp' and layer in layers: continue
                            donor=full if enabled else base
                            def freeze(act,hook,donor=donor):
                                act[:,-1,:]=donor[hook.name][:,-1,:]
                                return act
                            hooks.append((names[i],freeze))
                        observed=model.run_with_hooks(tokens,fwd_hooks=hooks)[0,-1]
                        error=(observed-logits[masks.index(mask)]).abs().max().item()
                        assert error<1e-4,(mode,mask,error)
                        controls[f'{mode}_{mask}']=error
                del logits,residual
            actual=[]
            if original is not None:
                actual=[original['records'][mask]['actual'] for mask in masks]
            else:
                for mask in masks:
                    hooks=[(first,add)]
                    for i,name in enumerate(names):
                        if (mask>>i)&1: continue
                        def freeze(act,hook):
                            act[:,-1,:]=base[hook.name][:,-1,:]
                            return act
                        hooks.append((name,freeze))
                    lg=model.run_with_hooks(tokens,fwd_hooks=hooks)[0,-1]
                    actual.append(metrics(lg)[0])
                    if mask in (0,255):
                        expected=fl if mask==255 else model.unembed(model.ln_final((base[final][0,-1]+vector)[None,None]))[0,0]
                        error=(lg-expected).abs().max().item()
                        assert error<1e-4,error
                        controls[f'endpoint_{mask}']=error
            return dict(prompt=prompt,answer=answer,baseline=metrics(bl)[0],full=metrics(fl)[0],controls=controls,
                records=[dict(mask=mask,actual=actual[i],predictions={mode:predictions[mode][i] for mode in MODES}) for i,mask in enumerate(masks)])
        with torch.inference_mode():
            for name in prior['cases']:
                original=json.loads((olddir/(name+'.json')).read_text())
                c=execute(f'The capital of {original["country"]} is',original['capital'],list(range(256)),original)
                c.update(country=original['country'],split=original['split'])
                atomic_json(OUT/(name+'.json'),c)
                manifest['original'].append(name)
                atomic_json(OUT/'run.json',manifest)
                print(f'{name}: three predictors × 256 masks; {time.monotonic()-start:.0f}s; {peak_mib():.0f} MiB',flush=True)
            for row in prior['scope_cases']:
                assert row['eligible']
                name=f'{row["form"]}__{row["country"]}'
                c=execute(row['prompt'],row['answer'],SCOPE_MASKS)
                c.update(country=row['country'],form=row['form'])
                for current,previous in [(c['baseline'],row['results']['baseline']),(c['full'],row['results']['capital'])]:
                    assert current['rank']==previous['rank'] and abs(current['p']-previous['p'])<1e-5
                atomic_json(OUT/(name+'.json'),c)
                manifest['scope'].append(name)
                atomic_json(OUT/'run.json',manifest)
            manifest.update(status='complete',elapsed_seconds=time.monotonic()-start,peak_rss_mib=peak_mib())
            atomic_json(OUT/'run.json',manifest)
            print(f'COMPLETE {manifest["elapsed_seconds"]:.1f}s; peak {peak_mib():.0f} MiB',flush=True)
    finally:
        done.set()
        lock.close()

if __name__=='__main__': main()
