"""Model-free verification of all SFT updates, tokens, initializations and costs."""
import argparse
import json
import math
from pathlib import Path
import random
import sys
import numpy as np
import stratified_protocol as sp
import stratified_models as sm
from training_ledger import verify_training_receipt
from verify_forward_ledger import verify as verify_ledger
from verify_budget_calibration import read
from verify_feasibility import ROOT,sha


def verify(out,require_checkpoints=False):
    m=read(out/'run.json');assert m['status']=='complete' and m['stage']=='finished'
    sp.require_provenance(m,ROOT);assert m['plan']==sp.PLAN_NAME
    name=m['target'];spec=sm.descriptor(name);native=name in sp.REFERENCES
    assert all(m[k]==v for k,v in spec.items()) and out.name==f'stratified-sft-{name}-v1'
    assert m['seed']==1220+sp.POPULATION.index(name)
    local=sm.local_checkpoint_hashes(name);assert m['local_checkpoint_hashes']==local
    assert m['cached_file_hashes']==sm.cached_file_hashes(name) and set(local).issubset(m['input_hashes'])
    missing=[]
    for path,digest in m['input_hashes'].items():
        if not (ROOT/path).exists() and path in local:missing.append(path)
        else:assert sha(ROOT/path)==digest,path
    for path,digest in m['output_hashes'].items():assert sha(out/path)==digest,path
    expected_outputs={'training-tokens.json','sft-training.json','sft-costs.json','training-ledger.json'}
    if native:expected_outputs.update(('initialization.json','initialization-logits.npz','initialization-ledger.json'))
    assert set(m['output_hashes'])==expected_outputs
    required={'fit_stratified_sft.py','verify_stratified_sft.py','run_stratified_sft.py','check_stratified_sft.py',
        'stratified_models.py','verify_stratified_behavior.py','verify_budget_behavior.py','verify_budget_calibration.py',
        'verify_stratified_calibration.py','elicitation.py','interventions.py','budget_inference.py','budget_protocol.py',
        'runtime.py','precision.py','feasibility.py','training_ledger.py','forward_ledger.py','verify_forward_ledger.py','verify_feasibility.py'}
    assert {'experiments/causal_audit/'+n for n in required}.issubset(m['input_hashes'])
    for member in sp.POPULATION:
        directory=ROOT/'data/causal_audit'/f'stratified-behavior-{member}-v1'
        assert {str((directory/p).relative_to(ROOT)) for p in ('run.json','selection.json')}.issubset(m['input_hashes'])
        behavior=read(directory/'run.json');assert behavior['status']=='complete' and behavior['target']==member
    sp.check_constructed([(n,read(ROOT/'data/causal_audit'/n/'run.json')) for n in sp.CONSTRUCTED])
    assert (m['dtype'],m['device'],m['attention_implementation'],m['training_padding'],m['adapter_key'])==('float32','mps','eager','right_dynamic','default')
    assert m['adapter_initialization']==('new_zero_output_lora' if native else 'continue_original_lora')
    assert m['limits']==dict(seconds=1800,rss_gib=32,mps_driver_gib=28,free_percent=15)
    assert 0<m['elapsed_seconds']<=1800 and m['peak_rss_gib']<=32 and m['peak_mps_driver_gib']<=28
    assert m['last_system_free_percent']>=15 and m['prerequisite_verification_seconds']>=0
    validation=read(ROOT/'data/causal_audit/development.json')['splits']['validation']['rows'];rows=validation[32:]
    ids=[r['id'] for r in rows];assert m['training_ids']==ids
    encoded=read(out/'training-tokens.json');assert [r['id'] for r in encoded]==ids
    choice_ids=m['choice_ids']
    for tokenized,row in zip(encoded,rows):
        assert tokenized['target_token_id']==choice_ids[row['answer']]
        assert 0<len(tokenized['input_ids'])<=512 and all(type(i) is int and 0<=i<151936 for i in tokenized['input_ids'])
    lengths={r['id']:len(r['input_ids']) for r in encoded}
    initialization=None
    if native:
        directory=ROOT/'data/causal_audit'/sp.REFERENCE_RUNS[name]
        if name=='reference-widened-base':directory=directory/'expanded'
        instrument_path=directory/'instrument-logits.npz'
        assert str(instrument_path.relative_to(ROOT)) in m['input_hashes']
        with np.load(instrument_path,allow_pickle=False) as arrays:reference=arrays['padded']
        with np.load(out/'initialization-logits.npz',allow_pickle=False) as arrays:
            assert arrays.files==['zero_adapter'];logits=arrays['zero_adapter']
        assert logits.shape==(4,151936) and logits.dtype==np.float32 and np.isfinite(logits).all()
        error=float(np.max(np.abs(logits-reference)))
        equal=bool(np.array_equal(logits[:,choice_ids].argmax(1),reference[:,choice_ids].argmax(1)))
        initialization=dict(ids=[r['id'] for r in validation[:4]],b_matrices=112,b_matrices_all_zero=True,
            trainable_parameters=6422528,max_full_logit_error=error,choice_predictions_equal=equal,
            passed=error<.001 and equal,forward_examples=4,forward_calls=1,padded_token_positions=2048)
        assert initialization==read(out/'initialization.json')==m['initialization'] and initialization['passed']
        ledger=read(out/'initialization-ledger.json');receipt=verify_ledger(ledger)
        assert receipt['completed_calls']==1 and receipt['attempted_examples_known']==4 and receipt['attempted_padded_input_tokens_known']==2048
        assert len(ledger['phases'])==1
        phase=ledger['phases'][0]
        assert phase['name']=='zero-adapter' and phase['expected_examples']==4 and phase['required_batch_size']==4 and phase['required_sequence_length']==512
        assert phase['require_inference'] is True and ledger['events'][0]['device'] in ('mps','mps:0')
        original=read(directory/'forward-ledger.json')
        assert phase['attempted_input_tokens']==original['events'][0]['input_tokens']
    else:assert m['initialization'] is None and m['initial_checkpoint_hashes']=={}
    training=read(out/'sft-training.json')
    assert training['seed']==m['seed'] and training['training_ids']==ids
    assert {k:training[k] for k in ('learning_rate','epochs','batch_size','weight_decay','gradient_clip','trainable_parameters')}==dict(
        learning_rate=1e-4,epochs=3,batch_size=4,weight_decay=.01,gradient_clip=1.,trainable_parameters=6422528)
    ledger=read(out/'training-ledger.json');receipt=verify_training_receipt(ledger)
    assert all(e['device'] in ('mps','mps:0') for e in ledger['events'])
    expected=[]
    for epoch in range(3):
        order=list(range(32));random.Random(m['seed']+epoch).shuffle(order)
        expected.extend((epoch+1,[ids[j] for j in order[i:i+4]]) for i in range(0,32,4))
    assert len(training['curve'])==len(ledger['events'])==24
    for step,(row,(epoch,batch),event) in enumerate(zip(training['curve'],expected,ledger['events']),1):
        assert row['step']==step and row['epoch']==epoch and row['batch_ids']==batch
        assert math.isfinite(row['loss']) and row['loss']>=0 and math.isfinite(row['gradient_norm']) and row['gradient_norm']>=0
        assert event['row_lengths']==[lengths[i] for i in batch]
    costs=read(out/'sft-costs.json');assert m['costs']==costs
    assert (costs['optimizer_steps'],costs['training_examples'],costs['unique_demonstrations'])==(24,96,32)
    assert costs['training_input_tokens']==3*sum(lengths.values())==sum(e['input_tokens'] for e in ledger['events'])
    assert 0<=training['elapsed_seconds']<=costs['seconds']<=m['elapsed_seconds']
    labels=('initial','sft') if native else ('sft',)
    for label in labels:
        hashes=m['initial_checkpoint_hashes' if label=='initial' else 'sft_checkpoint_hashes']
        assert {f'checkpoints/{label}/adapter_model.safetensors',f'checkpoints/{label}/adapter_config.json'}.issubset(hashes)
        for path,digest in hashes.items():
            if (out/path).exists():assert sha(out/path)==digest,path
            else:missing.append(str((out/path).relative_to(ROOT)))
        config=out/f'checkpoints/{label}/adapter_config.json'
        if config.exists():
            c=read(config);assert (c['r'],c['lora_alpha'],c['lora_dropout'],c['bias'],c['task_type'])==(16,32,.05,'none','CAUSAL_LM')
            assert set(c['target_modules'])=={'q_proj','k_proj','v_proj','o_proj'}
        weights=out/f'checkpoints/{label}/adapter_model.safetensors'
        if weights.exists():
            from safetensors.numpy import load_file
            tensors=load_file(weights)
            assert len(tensors)==224 and sum(v.size for v in tensors.values())==6422528
            assert all(v.dtype==np.float32 and np.isfinite(v).all() for v in tensors.values())
            a={n:v for n,v in tensors.items() if '.lora_A.' in n};b={n:v for n,v in tensors.items() if '.lora_B.' in n}
            assert len(a)==len(b)==112 and all(v.shape[0]==16 for v in a.values()) and all(v.shape[1]==16 for v in b.values())
            if label=='initial':assert all(np.count_nonzero(v)==0 for v in b.values())
    before=m['initial_checkpoint_hashes']['checkpoints/initial/adapter_model.safetensors'] if native else next(h for p,h in local.items() if p.endswith('/adapter_model.safetensors'))
    assert before!=m['sft_checkpoint_hashes']['checkpoints/sft/adapter_model.safetensors']
    if require_checkpoints:
        assert not missing,missing
        from verify_stratified_population import verify as verify_population
        assert verify_population(True)['verified']
    assert not any(n in sys.modules for n in ('torch','transformers','peft'))
    return dict(verified=True,target=name,seed=m['seed'],costs=costs,initialization=initialization,
        training_forward_receipt=receipt,checkpoint_files_unavailable=missing,model_loaded=False,
        scope='Recorded SFT and top-level presentations; no backward FLOP accounting or model rerun.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    parser.add_argument('--require-checkpoints',action='store_true');args=parser.parse_args()
    print(json.dumps(verify(args.run.resolve(),args.require_checkpoints),indent=2))


if __name__=='__main__':main()
