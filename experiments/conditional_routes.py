"""Exhaustive downstream intervention audit and fixed-vector scope test."""
from capital_generalization import ROOT, REVISION, atomic_json, watchdog, peak_mib
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import threading
import time

OUT = ROOT / 'data/conditional_routes'
PLAN = ROOT / 'notes/2026-09-05-conditional-route-plan.md'
SITES = [(l,b) for l in range(8,12) for b in ('attn','mlp')]
LANGUAGES = ['Danish','Norwegian','Swedish','Finnish','Polish','German','Hungarian','Dutch','Thai','Spanish']

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pf=json.loads((ROOT/'data/offset_mechanism/preflight.json').read_text())
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(Path.home()/f'.cache/huggingface/hub/models--gpt2/snapshots/{REVISION}/tokenizer.json'))
    scope=[]
    for (country,capital),language in zip(pf['replication'],LANGUAGES):
        for form,prompt,answer in [('possessive',f"{country}'s capital is",capital),
                ('qa',f'Question: What is the capital of {country}? Answer:',capital),
                ('language',f'The primary language of {country} is',language)]:
            ids=tokenizer.encode(' '+answer).ids
            scope.append(dict(country=country,capital=capital,form=form,prompt=prompt,answer=answer,
                              answer_tokens=ids,eligible=len(ids)==1))
    atomic_json(OUT/'preflight.json',scope)
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
        paths=[Path(__file__), PLAN, ROOT/'experiments/capital_generalization.py',
               ROOT/'data/offset_mechanism/offsets.npz',ROOT/'data/offset_mechanism/preflight.json',OUT/'preflight.json']
        manifest=dict(status='running',revision=REVISION,device='cpu',dtype='float32',
            hashes={str(p.relative_to(ROOT)):digest(p) for p in paths},
            versions={k:importlib.metadata.version(k) for k in ('torch','transformer-lens','transformers','numpy')},
            sites=SITES,cases=[],scope_cases=[])
        atomic_json(OUT/'run.json',manifest)
        arrays=np.load(ROOT/'data/offset_mechanism/offsets.npz')
        vectors={k:torch.from_numpy(arrays['resid_'+k].copy()) for k in ('capital','neutral','other_attribute')}
        model=HookedTransformer.from_pretrained('gpt2-small',device='cpu',dtype=torch.float32,
            local_files_only=True,revision=REVISION,default_prepend_bos=True)
        model.eval()
        names=[f'blocks.{l}.hook_{b}_out' for l,b in SITES]
        final='blocks.11.hook_resid_post'
        def add(act,hook):
            act[:,-1,:]+=vectors['capital']
            return act
        with torch.inference_mode():
            for split in ('replication','fresh'):
                for country,capital in pf[split]:
                    tokens=model.to_tokens(f'The capital of {country} is',prepend_bos=True)
                    bl,base=model.run_with_cache(tokens,names_filter=lambda n:n in names+[final])
                    bl=bl[0,-1].clone()
                    with model.hooks(fwd_hooks=[('blocks.8.hook_resid_pre',add)]):
                        fl,full=model.run_with_cache(tokens,names_filter=lambda n:n in names+[final])
                    fl=fl[0,-1].clone()
                    aid=model.to_single_token(' '+capital)
                    competitor=next(i for i in bl.topk(2).indices.tolist() if i!=aid)
                    def metric(logits):
                        return dict(rank=int((logits>logits[aid]).sum())+1,p=logits.softmax(-1)[aid].item(),
                            margin=(logits[aid]-logits[competitor]).item(),top_id=int(logits.argmax()),
                            top=model.to_string(int(logits.argmax())))
                    deltas=[full[n][0,-1]-base[n][0,-1] for n in names]
                    records=[]
                    controls={}
                    for mask in range(256):
                        hooks=[('blocks.8.hook_resid_pre',add)]
                        for i,n in enumerate(names):
                            if not (mask>>i)&1:
                                def freeze(act,hook):
                                    act[:,-1,:]=base[hook.name][:,-1,:]
                                    return act
                                hooks.append((n,freeze))
                        actual=model.run_with_hooks(tokens,fwd_hooks=hooks)[0,-1]
                        residual=base[final][0,-1]+vectors['capital']
                        for i,d in enumerate(deltas):
                            if (mask>>i)&1: residual=residual+d
                        predicted=model.unembed(model.ln_final(residual[None,None]))[0,0]
                        if mask in (0,255,85,170,87,171):
                            shadow_hooks=[('blocks.8.hook_resid_pre',add)]
                            for i,n in enumerate(names):
                                donor=full if (mask>>i)&1 else base
                                def fixed(act,hook,donor=donor):
                                    act[:,-1,:]=donor[hook.name][:,-1,:]
                                    return act
                                shadow_hooks.append((n,fixed))
                            shadow=model.run_with_hooks(tokens,fwd_hooks=shadow_hooks)[0,-1]
                            error=(shadow-predicted).abs().max().item()
                            assert error<1e-4,(country,mask,error)
                            controls[f'shadow_{mask}']=error
                        if mask==255:
                            error=(actual-fl).abs().max().item()
                            assert error<1e-4,error
                            controls['full_repeat']=error
                        records.append(dict(mask=mask,active=[f'{b}{l}' for i,(l,b) in enumerate(SITES) if (mask>>i)&1],
                            actual=metric(actual),predicted=metric(predicted),
                            kl=(actual.softmax(-1)*(actual.log_softmax(-1)-predicted.log_softmax(-1))).sum().item()))
                    # Explicit baseline donor self-patch, no offset.
                    def self_freeze(act,hook):
                        act[:,-1,:]=base[hook.name][:,-1,:]
                        return act
                    self_logits=model.run_with_hooks(tokens,fwd_hooks=[(n,self_freeze) for n in names])[0,-1]
                    controls['self_freeze']=(self_logits-bl).abs().max().item()
                    assert controls['self_freeze']<1e-4
                    old=json.loads((ROOT/f'data/offset_mechanism/{split}__{country}.json').read_text())
                    for key,new,prior in [('full',metric(fl),old['conditions']['capital']['full']),
                                          ('direct',records[0]['actual'],old['conditions']['capital']['direct'])]:
                        assert new['rank']==prior['rank']
                        controls[f'previous_{key}']=abs(new['p']-prior['p'])
                        assert controls[f'previous_{key}']<1e-5
                    case=dict(country=country,capital=capital,split=split,baseline=metric(bl),full=metric(fl),
                              controls=controls,records=records)
                    name=f'{split}__{country}'
                    atomic_json(OUT/f'{name}.json',case)
                    manifest['cases'].append(name)
                    atomic_json(OUT/'run.json',manifest)
                    disagreement=sum((r['actual']['rank']==1)!=(r['predicted']['rank']==1) for r in records)
                    print(f'{name}: 256 subsets; correctness disagreement {disagreement}/256; {time.monotonic()-start:.0f}s; {peak_mib():.0f} MiB',flush=True)
                    del base,full,bl,fl
            for row in scope:
                if not row['eligible']:
                    manifest['scope_cases'].append(row)
                    continue
                tokens=model.to_tokens(row['prompt'],prepend_bos=True)
                aid=model.to_single_token(' '+row['answer'])
                cid=model.to_single_token(' '+row['capital'])
                results={}
                for mode in ('baseline',*vectors):
                    def inject(act,hook,mode=mode):
                        act[:,-1,:]+=vectors[mode]
                        return act
                    hooks=[] if mode=='baseline' else [('blocks.8.hook_resid_pre',inject)]
                    logits=model.run_with_hooks(tokens,fwd_hooks=hooks)[0,-1]
                    results[mode]=dict(rank=int((logits>logits[aid]).sum())+1,p=logits.softmax(-1)[aid].item(),
                        top=model.to_string(int(logits.argmax())),capital_top=bool(logits.argmax()==cid),
                        capital_p=logits.softmax(-1)[cid].item())
                manifest['scope_cases'].append(dict(**row,results=results))
                atomic_json(OUT/'run.json',manifest)
            manifest.update(status='complete',elapsed_seconds=time.monotonic()-start,peak_rss_mib=peak_mib())
            atomic_json(OUT/'run.json',manifest)
            print(f'COMPLETE {manifest["elapsed_seconds"]:.1f}s; peak {peak_mib():.0f} MiB',flush=True)
    finally:
        done.set()
        lock.close()

if __name__=='__main__': main()
