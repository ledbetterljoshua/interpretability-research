"""Reconstruct the widening preflight from saved evidence, without any LM."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from runtime import ROOT, sha
from verify_feasibility import finite
from verify_forward_ledger import verify as verify_ledger
from budget_instrument_verification import verify as verify_instrument
from reference_preflight_validation import evaluation
from widening_preflight_validation import compare, check_weights
from prepare_widening_reference import MODEL, REVISION, FILES
import widening
import budget_protocol as bp
import interventions as it
from reference_format import ReferenceTokenizer

LABELS=('ordinary','few_shot','source_code')


def read(path):
    result=json.loads(path.read_text());finite(result);return result


def verify_state(out,state,config,rows,cases,choice_ids,pad_id):
    directory=out/state
    arrays=dict(np.load(directory/'captures.npz',allow_pickle=False))
    assert set(arrays)=={'logits','residuals','input_ids','attention_mask'}
    assert arrays['logits'].shape==(24,151936)
    assert arrays['residuals'].shape==(28,24,config['hidden_size'])
    assert all(arrays[k].dtype==np.float32 and np.isfinite(arrays[k]).all() for k in ('logits','residuals'))
    assert arrays['input_ids'].shape==arrays['attention_mask'].shape==(24,512)
    assert all(arrays[k].dtype==np.int64 for k in ('input_ids','attention_mask'))
    for i,case in enumerate(cases):
        tokens=case['input_ids'];n=len(tokens);assert 0<n<=512
        assert arrays['input_ids'][i].tolist()==[pad_id]*(512-n)+tokens
        assert arrays['attention_mask'][i].tolist()==[0]*(512-n)+[1]*n
    inventory=read(directory/'inventory.json')
    assert inventory==dict(unique_tensors=310,parameters=596049920 if state=='native' else 1720574976,tied_embeddings=True)
    ledger=read(directory/'forward-ledger.json');receipt=verify_ledger(ledger)
    assert receipt['completed_calls']==14 and receipt['attempted_examples_known']==44
    assert receipt['attempted_padded_input_tokens_known']==40*512+sum(len(c['input_ids']) for c in cases[:4])
    assert all(e['device'] in ('mps','mps:0') for e in ledger['events'])
    phases={p['name']:p for p in ledger['phases']};assert list(phases)==['instruments',*LABELS]
    instrument=read(directory/'instruments.json')
    raw=dict(np.load(directory/'instrument-logits.npz',allow_pickle=False))
    numerical=verify_instrument(instrument,raw,phases['instruments'],ledger['events'],[r['id'] for r in rows[:4]],choice_ids)
    assert numerical['ordinary_prompt_lengths']==[len(c['input_ids']) for c in cases[:4]]
    # This is the same ordinary batch, independently repeated by the policy pass.
    assert np.max(np.abs(raw['padded']-arrays['logits'][:4]))<.001
    correct={}
    for j,label in enumerate(LABELS):
        phase=phases[label];subset=cases[8*j:8*j+8]
        assert phase['expected_examples']==phase['completed_examples']==8 and phase['completed_calls']==2
        assert phase['required_batch_size']==4 and phase['required_sequence_length']==512
        assert phase['attempted_input_tokens']==sum(len(c['input_ids']) for c in subset)
        for k,index in enumerate(phase['event_indices']):
            assert ledger['events'][index]['input_tokens']==sum(len(c['input_ids']) for c in subset[4*k:4*k+4])
        result=evaluation(directory,label,rows,choice_ids,phase)
        correct[label]=result['correct']
        for i,r in enumerate(result['records']):
            z=arrays['logits'][8*j+i];assert r['choice_logits']==z[choice_ids].tolist()
            assert r['top_token_id']==int(z.argmax())
            exp=np.exp(z.astype(np.float64)-float(z.max()))
            mass=float(exp[choice_ids].sum()/exp.sum())
            assert abs(mass-r['choice_mass'])<2e-6
    return arrays,dict(correct=correct,headroom={k:1-v/8 for k,v in correct.items()},instruments=numerical,
                       forward_examples=44,calls=14,padded_input_positions=receipt['attempted_padded_input_tokens_known'])


def verify(out,require_checkpoints=False,cache_root=None):
    m=read(out/'run.json')
    assert m['status'] in ('complete','error') and m['stage']=='finished-equivalence'
    assert (m['model'],m['revision'])==(MODEL,REVISION)
    assert (m['dtype'],m['device'],m['attention_implementation'],m['training'])==('float32','mps','eager',False)
    expected_modules={'widening_preflight.py','verify_widening_preflight.py','widening_preflight_validation.py',
        'check_widening_preflight.py','widening.py','runtime.py','forward_ledger.py','budget_inference.py',
        'budget_protocol.py','interventions.py','reference_format.py','prepare_widening_reference.py',
        'inspect_widening_configs.py','check_widening.py','verify_lower_gold_replication.py',
        'verify_forward_ledger.py','budget_instrument_verification.py','reference_preflight_validation.py','verify_feasibility.py'}
    assert {Path(n).name for n in m['input_hashes'] if n.startswith('experiments/')}==expected_modules
    required={'notes/2026-09-12-causal-audit-widening-preflight-plan.md','data/causal_audit/development.json',
        'data/causal_audit/widening-reference-download-v1/download.json',
        'data/causal_audit/widening-config-inspection-v1/inspection.json',
        'data/causal_audit/widening-config-inspection-v1/small_base-config.json',
        'data/causal_audit/widening-config-inspection-v1/large_base-config.json',
        'data/causal_audit/widening-algebra-check-v1.json',
        *[f'data/causal_audit/lower-gold-{a}-1289-v1/run.json' for a in ('conditional','marginal')]}
    assert set(m['input_hashes'])==required|{'experiments/causal_audit/'+n for n in expected_modules}
    for name,digest in m['input_hashes'].items():assert sha(ROOT/name)==digest,name
    for arm in ('conditional','marginal'):
        assert read(ROOT/f'data/causal_audit/lower-gold-{arm}-1289-v1/run.json')['status']=='complete'
    expected_outputs={'tokenization.json','weight-verification.json','summary.json'}|{
        state+'/'+name for state in ('native','expanded') for name in ('inventory.json','instrument-logits.npz',
            'instruments.json','ordinary.json','few_shot.json','source_code.json','captures.npz','forward-ledger.json')}
    assert set(m['output_hashes'])==expected_outputs
    for name,digest in m['output_hashes'].items():assert sha(out/name)==digest,name
    assert m['limits']==dict(seconds=1800,rss_gib=32,mps_driver_gib=28,free_percent=15)
    assert 0<m['elapsed_seconds']<=1800 and m['peak_rss_gib']<=32 and m['peak_mps_driver_gib']<=28
    assert m['last_system_free_percent']>=15
    download=read(ROOT/'data/causal_audit/widening-reference-download-v1/download.json')
    assert m['cached_file_hashes']==download['cached_file_hashes'] and set(m['cached_file_hashes'])==set(FILES)
    small=read(ROOT/'data/causal_audit/widening-config-inspection-v1/small_base-config.json')
    large=read(ROOT/'data/causal_audit/widening-config-inspection-v1/large_base-config.json');widening.validate_configs(small,large)
    d=read(ROOT/'data/causal_audit/development.json')['splits'];rows=d['validation']['rows'][:8]
    canonical=[next(r for r in d['train']['rows'] if r['answer']==i) for i in range(4)]
    tokenization=read(out/'tokenization.json');cases=tokenization['cases']
    assert len(cases)==24 and tokenization['choice_ids']==[362,425,356,422]
    assert tokenization['demonstration_ids']==[r['id'] for r in canonical]
    assert type(tokenization['pad_token_id']) is int and 0<=tokenization['pad_token_id']<151936
    for j,label in enumerate(LABELS):
        policy=next(p for p in bp.BASELINE_POLICIES if p['name']==label)
        for r,c in zip(rows,cases[j*8:j*8+8]):
            assert (c['id'],c['policy'])==(r['id'],label)
            assert c['prompt']==it.prompt(ReferenceTokenizer(None,'base'),r,policy['prefix'],bp.examples(policy,canonical))
            assert all(type(t) is int and 0<=t<151936 for t in c['input_ids'])
    a,na=verify_state(out,'native',small,rows,cases,tokenization['choice_ids'],tokenization['pad_token_id'])
    b,nb=verify_state(out,'expanded',large,rows,cases,tokenization['choice_ids'],tokenization['pad_token_id'])
    assert all(np.array_equal(a[k],b[k]) for k in ('input_ids','attention_mask'))
    result=compare(a,b,tokenization['choice_ids'])
    result.update(instruments_passed=True,tied_embeddings=True,weight_transformation_verified=True)
    assert result==read(out/'summary.json')==m['summary']
    assert (m['status']=='complete')==result['ready']
    if not result['ready']:assert m['error']=='AssertionError: Fixed numerical-equivalence forecast failed'
    expected_weights=dict(unique_tensors=310,native_matches_pinned_bf16=True,
        expanded_exactly_matches_transformation=True,dtype='float32',finite=True,
        native_parameters=596049920,expanded_parameters=1720574976)
    assert read(out/'weight-verification.json')==expected_weights
    expected_checkpoints={'checkpoints/native-effective/config.json','checkpoints/native-effective/generation_config.json',
        'checkpoints/native-effective/model.safetensors','checkpoints/expanded/config.json','checkpoints/expanded/model.safetensors'}
    assert set(m['checkpoint_hashes'])==expected_checkpoints
    missing=[name for name in expected_checkpoints if not (out/name).exists()]
    if require_checkpoints:
        assert not missing,missing
        for name,digest in m['checkpoint_hashes'].items():assert sha(out/name)==digest,name
        from huggingface_hub.constants import HF_HUB_CACHE
        cache=Path(cache_root or HF_HUB_CACHE)/('models--'+MODEL.replace('/','--'))/'snapshots'/REVISION
        for name,digest in m['cached_file_hashes'].items():assert sha(cache/name)==digest,name
        assert sha(out/'checkpoints/expanded/config.json')==sha(ROOT/'data/causal_audit/widening-config-inspection-v1/large_base-config.json')
        assert check_weights(out/'checkpoints/native-effective/model.safetensors',out/'checkpoints/expanded/model.safetensors',
            cache/'model.safetensors',small,large)==expected_weights
        from tokenizers import Tokenizer
        tokenizer=Tokenizer.from_file(str(cache/'tokenizer.json'));tokenizer.no_padding();tokenizer.no_truncation()
        assert [tokenizer.encode(' '+c,add_special_tokens=False).ids for c in 'ABCD']==[[t] for t in tokenization['choice_ids']]
        config=read(cache/'tokenizer_config.json')
        assert tokenizer.token_to_id(config['pad_token'])==tokenization['pad_token_id']
        for case in cases:assert tokenizer.encode(case['prompt']).ids==case['input_ids']
    assert not any(n in sys.modules for n in ('torch','transformers','peft'))
    return dict(verified=True,ready=result['ready'],status=m['status'],summary=result,native=na,expanded=nb,
        forward_examples=88,calls=28,checkpoint_files_unavailable=missing,
        checkpoint_transformation_rechecked=require_checkpoints,tokenization_rechecked=require_checkpoints,
        model_loaded=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    parser.add_argument('--require-checkpoints',action='store_true');parser.add_argument('--cache-root',type=Path)
    parser.add_argument('--require-ready',action='store_true');args=parser.parse_args()
    result=verify(args.run.resolve(),args.require_checkpoints,args.cache_root)
    print(json.dumps(result,indent=2))
    if args.require_ready:assert result['ready'],'Numerical-equivalence forecast failed'


if __name__=='__main__':main()
