"""Prespecified sequential native/expanded inference; no training or test data."""
from runtime import ROOT, Run, atomic_json, configure, sha
configure()
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
import weakref
import numpy as np
import budget_inference as bi
import budget_protocol as bp
import interventions as it
import reference_format as rf
import widening
from forward_ledger import ForwardLedger
from prepare_widening_reference import MODEL, REVISION, FILES
from widening_preflight_validation import compare, check_weights

PLAN = ROOT/'notes/2026-09-12-causal-audit-widening-preflight-plan.md'
OUT = ROOT/'data/causal_audit/widening-preflight-base-v1'
DATA = ROOT/'data/causal_audit/development.json'
CONFIGS = ROOT/'data/causal_audit/widening-config-inspection-v1'
DOWNLOAD = ROOT/'data/causal_audit/widening-reference-download-v1/download.json'
LABELS = ('ordinary','few_shot','source_code')
MODULES = ('widening_preflight.py','verify_widening_preflight.py','widening_preflight_validation.py',
    'check_widening_preflight.py','widening.py','runtime.py','forward_ledger.py','budget_inference.py',
    'budget_protocol.py','interventions.py','reference_format.py','prepare_widening_reference.py',
    'inspect_widening_configs.py','check_widening.py','verify_lower_gold_replication.py',
    'verify_forward_ledger.py','budget_instrument_verification.py','reference_preflight_validation.py',
    'verify_feasibility.py')


def prerequisites():
    start = time.monotonic()
    pair = [ROOT/f'data/causal_audit/lower-gold-{arm}-1289-v1/run.json' for arm in ('conditional','marginal')]
    assert all(p.exists() and json.loads(p.read_text())['status']=='complete' for p in pair)
    scripts = [( 'verify_lower_gold_replication.py',*[str(p.parent) for p in pair],'--require-pair','--require-checkpoints'),
        ('prepare_widening_reference.py','--verify'),('inspect_widening_configs.py','--verify'),('check_widening.py','--verify')]
    receipts = []
    for script,*args in scripts:
        result = subprocess.run([sys.executable,str(Path(__file__).with_name(script)),*args],
            check=True,capture_output=True,text=True,timeout=300)
        receipts.append(dict(script=script,result=json.loads(result.stdout)))
    sources = [*[Path(__file__).with_name(n) for n in MODULES],DATA,DOWNLOAD,*pair,
        CONFIGS/'inspection.json',CONFIGS/'small_base-config.json',CONFIGS/'large_base-config.json',
        ROOT/'data/causal_audit/widening-algebra-check-v1.json']
    for p in [PLAN,*sources]:
        assert subprocess.check_output(['git','show',f'HEAD:{p.relative_to(ROOT)}'],cwd=ROOT)==p.read_bytes(),p
    return sources,dict(seconds=time.monotonic()-start,checks=receipts)


def evaluate_state(path, config, tokenizer, rows, canonical, run, state):
    import torch
    from transformers import AutoModelForCausalLM
    out = OUT/state; out.mkdir()
    run.save(stage=f'{state}-loading')
    model = AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.float32,
        attn_implementation='eager').to('mps').eval()
    ref = weakref.ref(model)
    names = dict(model.named_parameters())
    assert set(names)==set(widening.shapes(config))
    assert all(tuple(v.shape)==widening.shapes(config)[n] and v.dtype==torch.float32 for n,v in names.items())
    tied = model.lm_head.weight is model.model.embed_tokens.weight
    assert tied and model.lm_head.weight.data_ptr()==model.model.embed_tokens.weight.data_ptr()
    inventory = dict(unique_tensors=len(names),parameters=sum(v.numel() for v in names.values()),tied_embeddings=tied)
    del names
    atomic_json(out/'inventory.json',inventory)
    choice_ids = tokenizer.choice_ids(); assert choice_ids==[362,425,356,422]
    collected = dict(logits=[],residuals=[[] for _ in range(config['num_hidden_layers'])],input_ids=[],attention_mask=[])
    def capture_output(module,args,kwargs,output):
        collected['logits'].append(output.logits[:,-1].detach().float().cpu().numpy().copy())
        for key in ('input_ids','attention_mask'):
            collected[key].append(kwargs[key].detach().cpu().numpy().copy())
    def block_hook(index):
        def capture(module,args,output):
            collected['residuals'][index].append(output[:,-1].detach().float().cpu().numpy().copy())
        return capture
    with ForwardLedger(model) as ledger:
        try:
            run.save(stage=f'{state}-instruments')
            raw = {}
            with ledger.phase('instruments',expected_examples=20):
                instrument = bi.check_instruments(model,tokenizer,rows[:4],choice_ids,full_logits_out=raw)
            np.savez_compressed(out/'instrument-logits.npz',**{k:v.numpy() for k,v in raw.items()})
            atomic_json(out/'instruments.json',instrument)
            del raw
            assert instrument['passed'],'Numerical instrument forecast failed'
            handles = [model.register_forward_hook(capture_output,with_kwargs=True)]
            handles += [block.register_forward_hook(block_hook(i)) for i,block in enumerate(it.decoder_layers(model))]
            try:
                for label in LABELS:
                    policy=next(p for p in bp.BASELINE_POLICIES if p['name']==label)
                    run.save(stage=f'{state}-{label}')
                    with ledger.phase(label,expected_examples=8,sequence_length=512,batch_size=4):
                        result=bi.evaluate(model,tokenizer,rows,choice_ids,policy['prefix'],bp.examples(policy,canonical),label)
                    atomic_json(out/f'{label}.json',result)
            finally:
                for h in handles:h.remove()
            arrays = {k:np.concatenate(collected[k]) for k in ('logits','input_ids','attention_mask')}
            arrays['residuals']=np.stack([np.concatenate(v) for v in collected['residuals']])
            assert all(np.isfinite(v).all() for v in arrays.values())
            np.savez_compressed(out/'captures.npz',**arrays)
            if state=='native':
                run.save(stage='saving-native-effective-weights')
                model.save_pretrained(OUT/'checkpoints/native-effective',max_shard_size='10GB')
        finally:
            atomic_json(out/'forward-ledger.json',ledger.snapshot())
    del ledger,model
    gc.collect();torch.mps.empty_cache()
    assert ref() is None,'Native model still held alive'
    return arrays,inventory


def convert(small, large):
    from safetensors import safe_open
    from safetensors.numpy import save_file
    native=OUT/'checkpoints/native-effective/model.safetensors'
    destination=OUT/'checkpoints/expanded';destination.mkdir(parents=True)
    with safe_open(native,framework='np') as source:
        assert set(source.keys())==set(widening.shapes(small))
        expanded={n:widening.widen_weight(n,source.get_tensor(n),small,large) for n in source.keys()}
        assert all(v.dtype==np.float32 and np.isfinite(v).all() for v in expanded.values())
        save_file(expanded,destination/'model.safetensors',metadata={'format':'pt'})
    (destination/'config.json').write_bytes((CONFIGS/'large_base-config.json').read_bytes())
    del expanded
    gc.collect()


def main():
    sources,prerequisite_receipt=prerequisites()
    from huggingface_hub.constants import HF_HUB_CACHE
    cache=Path(HF_HUB_CACHE)/('models--'+MODEL.replace('/','--'))/'snapshots'/REVISION
    small=json.loads((CONFIGS/'small_base-config.json').read_text())
    large=json.loads((CONFIGS/'large_base-config.json').read_text())
    widening.validate_configs(small,large)
    d=json.loads(DATA.read_text())['splits'];rows=d['validation']['rows'][:8]
    canonical=[next(r for r in d['train']['rows'] if r['answer']==i) for i in range(4)]
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(1429)
    with Run(OUT,PLAN,sources,seconds=1800) as run:
        try:
            tokenizer=rf.ReferenceTokenizer(AutoTokenizer.from_pretrained(cache,local_files_only=True),'base')
            cases=[]
            for label in LABELS:
                p=next(p for p in bp.BASELINE_POLICIES if p['name']==label)
                for r in rows:
                    prompt=it.prompt(tokenizer,r,p['prefix'],bp.examples(p,canonical))
                    tokens=tokenizer.encode(prompt)
                    assert 0<len(tokens)<=512 and all(0<=i<small['vocab_size'] for i in tokens)
                    cases.append(dict(policy=label,id=r['id'],prompt=prompt,input_ids=tokens))
            atomic_json(OUT/'tokenization.json',dict(cases=cases,pad_token_id=tokenizer.pad_token_id,
                choice_ids=tokenizer.choice_ids(),demonstration_ids=[r['id'] for r in canonical]))
            run.save(stage='prepared',model=MODEL,revision=REVISION,dtype='float32',device='mps',
                attention_implementation='eager',training=False,prerequisites=prerequisite_receipt,
                cached_snapshot=str(cache),cached_file_hashes=json.loads(DOWNLOAD.read_text())['cached_file_hashes'])
            a,ia=evaluate_state(cache,small,tokenizer,rows,canonical,run,'native')
            run.save(stage='converting-weights')
            convert(small,large)
            run.save(stage='verifying-every-transformed-weight')
            weights=check_weights(OUT/'checkpoints/native-effective/model.safetensors',
                OUT/'checkpoints/expanded/model.safetensors',cache/'model.safetensors',small,large)
            atomic_json(OUT/'weight-verification.json',weights)
            b,ib=evaluate_state(OUT/'checkpoints/expanded',large,tokenizer,rows,canonical,run,'expanded')
            assert all(np.array_equal(a[k],b[k]) for k in ('input_ids','attention_mask'))
            result=compare(a,b,tokenizer.choice_ids())
            result['instruments_passed']=True
            result['tied_embeddings']=ia['tied_embeddings'] and ib['tied_embeddings']
            result['weight_transformation_verified']=True
            atomic_json(OUT/'summary.json',result)
            run.save(stage='finished-equivalence',summary=result)
            assert result['ready'],'Fixed numerical-equivalence forecast failed'
            print(json.dumps(result,indent=2),flush=True)
        finally:
            run.save(output_hashes={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*'))
                if p.is_file() and p.name!='run.json' and 'checkpoints' not in p.relative_to(OUT).parts},
                checkpoint_hashes={str(p.relative_to(OUT)):sha(p) for p in sorted((OUT/'checkpoints').rglob('*')) if p.is_file()})


if __name__=='__main__':main()
