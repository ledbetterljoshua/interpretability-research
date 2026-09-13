"""Independently reconstruct all behavioral candidates and the fixed budget."""
import argparse
import json
from pathlib import Path
import sys
import stratified_protocol as sp
import stratified_models as sm
from budget_protocol import BASELINE_POLICIES
from verify_budget_behavior import check_selection
from verify_budget_calibration import read,evaluation
from verify_feasibility import ROOT,sha
from verify_forward_ledger import verify as verify_ledger


def verify(out,require_checkpoints=False):
    m=read(out/'run.json');assert m['status']=='complete' and m['stage']=='finished'
    sp.require_provenance(m,ROOT);assert m['plan']==sp.PLAN_NAME
    name=m['target'];assert name in sp.POPULATION
    assert out.name==f'stratified-behavior-{name}-v1'
    spec=sm.descriptor(name);assert all(m[k]==v for k,v in spec.items())
    local=sm.local_checkpoint_hashes(name)
    assert m['local_checkpoint_hashes']==local and m['cached_file_hashes']==sm.cached_file_hashes(name)
    assert set(local).issubset(m['input_hashes'])
    missing=[]
    for path,digest in m['input_hashes'].items():
        if not (ROOT/path).exists() and path in local:missing.append(path)
        else:assert sha(ROOT/path)==digest,path
    if require_checkpoints:
        assert not missing,missing
        # Reconstruct all native/cache and original-constructed prerequisites.
        from verify_stratified_population import verify as verify_population
        assert verify_population(True)['verified']
    required={'fit_stratified_behavior.py','verify_stratified_behavior.py','stratified_models.py',
        'run_stratified_behavior.py','check_stratified_behavior.py',
        'budget_inference.py','budget_protocol.py','budget_selection.py','score_calibration.py','interventions.py',
        'runtime.py','forward_ledger.py','verify_forward_ledger.py','verify_stratified_calibration.py',
        'verify_budget_behavior.py','verify_budget_calibration.py','verify_feasibility.py'}
    assert {'experiments/causal_audit/'+n for n in required}.issubset(m['input_hashes'])
    assert {'data/causal_audit/stratified-calibration-v1/run.json',
        'data/causal_audit/stratified-calibration-v1/selection.json','data/causal_audit/development.json'}.issubset(m['input_hashes'])
    for path,digest in m['output_hashes'].items():assert sha(out/path)==digest,path
    labels=[p['name'] for p in BASELINE_POLICIES]
    assert set(m['output_hashes'])=={'selection.json','forward-ledger.json',*[f'policy-{label}.json' for label in labels]}
    assert (m['dtype'],m['device'],m['attention_implementation'],m['padding_length'],m['batch_size'])==('float32','mps','eager',512,4)
    assert m['policies']==BASELINE_POLICIES
    assert m['limits']==dict(seconds=1800,rss_gib=32,mps_driver_gib=28,free_percent=15)
    assert 0<m['elapsed_seconds']<=1800 and m['peak_rss_gib']<=32 and m['peak_mps_driver_gib']<=28
    assert m['last_system_free_percent']>=15 and m['prerequisite_verification_seconds']>=0
    sp.check_constructed([(n,read(ROOT/'data/causal_audit'/n/'run.json')) for n in sp.CONSTRUCTED])
    d=read(ROOT/'data/causal_audit/development.json')['splits'];rows=d['validation']['rows'][32:]
    assert m['selection_ids']==[r['id'] for r in rows]
    assert m['demonstration_ids']==[next(r['id'] for r in d['train']['rows'] if r['answer']==i) for i in range(4)]
    ledger=read(out/'forward-ledger.json');receipt=verify_ledger(ledger)
    assert receipt['completed_calls']==176 and receipt['attempted_examples_known']==704
    assert receipt['attempted_padded_input_tokens_known']==360448
    assert [p['name'] for p in ledger['phases']]==labels
    assert all(e['device'] in ('mps','mps:0') for e in ledger['events'])
    evaluations=[]
    for phase in ledger['phases']:
        assert phase['completed_examples']==phase['expected_examples']==32 and phase['completed_calls']==8
        assert phase['required_batch_size']==4 and phase['required_sequence_length']==512
        evaluations.append(evaluation(out,phase['name'],rows,m['choice_ids'],phase,filename=f"policy-{phase['name']}.json"))
    selected=read(out/'selection.json');check_selection(selected,evaluations,rows)
    assert m['prompt_only']==selected['prompt_only'] and m['decoded']==selected['decoded']
    assert m['cpu_selection_seconds']==selected['cpu_seconds']
    assert not any(n in sys.modules for n in ('torch','transformers','peft'))
    return dict(verified=True,target=name,candidate_count=616,forward_counts=receipt,
        prompt_only=selected['prompt_only'],decoded=selected['decoded'],checkpoint_files_unavailable=missing,
        native_cache_and_checkpoints_rechecked=require_checkpoints,model_loaded=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    parser.add_argument('--require-checkpoints',action='store_true');args=parser.parse_args()
    print(json.dumps(verify(args.run.resolve(),args.require_checkpoints),indent=2))


if __name__=='__main__':main()
