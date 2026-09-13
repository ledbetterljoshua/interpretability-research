"""Reconstruct held-out scored views, diagnostics and actual calls without an LM."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import stratified_protocol as sp
import stratified_models as sm
from stratified_freeze import MODULES,DATA,HOLDOUT,HOLDOUT_SHA,CALIBRATION,DIRECTIONS,SPLITS,fitting_stages,require_complete_fits
from verify_feasibility import ROOT,sha
from verify_budget_calibration import read,evaluation
from verify_forward_ledger import verify as verify_ledger
from budget_instrument_verification import verify as verify_instrument
from verify_budget_test import predictions,IDENTITY


def verify(out,require_checkpoints=False):
    m=read(out/'run.json');assert m['status']=='complete' and m['stage']=='finished'
    sp.require_provenance(m,ROOT);assert m['plan']==sp.PLAN_NAME
    name=m['target'];spec=sm.descriptor(name);native=name in sp.REFERENCES;widened=name=='reference-widened-base'
    source=read(CALIBRATION/'selection.json');assert m['source_selection']==source
    assert out.name==f'stratified-test-{name}-v1' and all(m[k]==v for k,v in spec.items())
    sft=ROOT/'data/causal_audit'/f'stratified-sft-{name}-v1';sft_manifest=read(sft/'run.json')
    local=sm.local_checkpoint_hashes(name)
    local.update({str((sft/n).relative_to(ROOT)):h for n,h in sft_manifest['sft_checkpoint_hashes'].items()})
    assert m['local_checkpoint_hashes']==local and m['cached_file_hashes']==sm.cached_file_hashes(name)
    assert set(local).issubset(m['input_hashes'])
    assert all(m['input_hashes'][n]==h for n,h in local.items())
    missing=[]
    for path,digest in m['input_hashes'].items():
        if not (ROOT/path).exists() and path in local:missing.append(path)
        else:assert sha(ROOT/path)==digest,path
    assert {'experiments/causal_audit/'+n for n in MODULES}.issubset(m['input_hashes'])
    required={str(p.relative_to(ROOT)) for p in (DATA,HOLDOUT,CALIBRATION/'selection.json',CALIBRATION/'vectors.npz')}
    if not source['abstain']:
        required.update(str(p.relative_to(ROOT)) for p in (DIRECTIONS/'directions.json',DIRECTIONS/'directions.npz'))
    stages=fitting_stages();require_complete_fits(stages)
    required.update(str((p/'run.json').relative_to(ROOT)) for s,n,p in stages)
    required.update(str((p/'selection.json').relative_to(ROOT)) for s,n,p in stages if s in ('calibration','behavior'))
    assert required.issubset(m['input_hashes'])
    if require_checkpoints:
        assert not missing,missing
        from verify_stratified_population import verify as verify_population
        from verify_stratified_sft import verify as verify_sft
        assert verify_population(True)['verified']
        r=verify_sft(sft);assert r['verified'] and not r['checkpoint_files_unavailable']
    for path,digest in m['output_hashes'].items():assert sha(out/path)==digest,path
    assert (m['dtype'],m['device'],m['attention_implementation'],m['padding_length'],m['batch_size'])==('float32','mps','eager',512,4)
    assert m['limits']==dict(seconds=3600,rss_gib=32,mps_driver_gib=28,free_percent=15)
    assert 0<m['elapsed_seconds']<=3600 and m['peak_rss_gib']<=32 and m['peak_mps_driver_gib']<=28
    assert m['last_system_free_percent']>=15 and m['prerequisite_verification_seconds']>=0
    assert sha(HOLDOUT)==HOLDOUT_SHA
    splits=read(HOLDOUT)['splits'];assert set(splits)==set(SPLITS)
    assert all(len(splits[s]['rows'])==256 for s in SPLITS)
    assert m['test_ids']=={s:[r['id'] for r in splits[s]['rows']] for s in SPLITS}
    assert m['excluded']=={s:[] for s in SPLITS}
    dev=read(DATA)['splits'];ids=[r['id'] for r in dev['validation']['rows'][:4]]
    assert m['instrument_ids']==ids
    assert m['demonstration_ids']==[next(r['id'] for r in dev['train']['rows'] if r['answer']==i) for i in range(4)]
    behavior=read(ROOT/'data/causal_audit'/f'stratified-behavior-{name}-v1'/'selection.json')
    assert m['behavioral_winners']=={k:behavior[k] for k in ('prompt_only','decoded')}
    own=None if native else read(ROOT/'data/causal_audit'/name/'run.json')['conditions']['unlock']
    assert m['own_code_prefix']==own and m['graft_diagnostics_skipped']==source['abstain']
    assert m['random_direction_seeds']==[1215,1216,1217]
    projection=None if source['abstain'] else read(DIRECTIONS/'directions.json')
    assert m['widened_projection']==(projection['projection'] if widened and not source['abstain'] else None)
    projection_run=widened and not source['abstain'] and projection['projection']['nondegenerate']
    assert m['widened_projection_skipped']==(not projection_run)
    if widened and not source['abstain']:
        from prepare_stratified_widened_diagnostics import reconstruct
        arrays,expected=reconstruct()
        assert {k:v for k,v in projection.items() if k!='array_sha256'}==expected
        assert projection['array_sha256']==sha(DIRECTIONS/'directions.npz')
        with np.load(DIRECTIONS/'directions.npz',allow_pickle=False) as saved:
            assert set(saved.files)==set(arrays)
            for k in arrays:assert np.array_equal(saved[k],arrays[k])
    sp.check_constructed([(n,read(ROOT/'data/causal_audit'/n/'run.json')) for n in sp.CONSTRUCTED])
    ledger=read(out/'forward-ledger.json');receipt=verify_ledger(ledger)
    assert all(e['device'] in ('mps','mps:0') for e in ledger['events'])
    phases={p['name']:p for p in ledger['phases']};expected_phases=[]
    expected_outputs={'forward-ledger.json','summary.json','random-writes.npz'};table=[];instruments=[]

    def instrument(label):
        phase='instrument-'+label;expected_phases.append(phase)
        expected_outputs.update((phase+'.json',phase+'.npz'))
        with np.load(out/(phase+'.npz'),allow_pickle=False) as arrays:
            instruments.append(verify_instrument(read(out/(phase+'.json')),dict(arrays),phases[phase],ledger['events'],ids,m['choice_ids']))

    def forward(split,label):
        phase=f'{split}/{label}';expected_phases.append(phase)
        filename=f'{split}/forward-{label}.json';expected_outputs.add(filename)
        p=phases[phase]
        assert p['completed_examples']==p['expected_examples']==256 and p['completed_calls']==64
        assert p['required_batch_size']==4 and p['required_sequence_length']==512 and p['require_inference'] is True
        return evaluation(out,phase,splits[split]['rows'],m['choice_ids'],p,filename=filename),filename

    def view(split,label,pair,decoder=IDENTITY,abstained=False):
        raw,filename=pair;path=f'{split}/method-{label}.json';expected_outputs.add(path)
        result=read(out/path)
        assert (result['method'],result['n'],result['source_evaluation'],result['abstained'],result['additional_model_forwards'])==(label,256,filename,abstained,0)
        assert result['decoder']==decoder
        expected=[dict(id=r['id'],answer=r['answer'],prediction=p) for r,p in zip(raw['records'],predictions(raw['records'],decoder))]
        assert result['records']==expected
        correct=sum(r['prediction']==r['answer'] for r in expected);assert result['correct']==correct
        table.append(dict(split=split,method=label,n=256,correct=correct,abstained=abstained))

    with np.load(out/'random-writes.npz',allow_pickle=False) as arrays:
        assert arrays.files==['directions'];directions=arrays['directions']
    assert directions.shape==(3,2048) and directions.dtype==np.float32
    for i,seed in enumerate((1215,1216,1217)):
        v=np.random.default_rng(seed).standard_normal(2048);v/=np.linalg.norm(v)
        assert np.array_equal(directions[i],v.astype(np.float32))
    policies=list(dict.fromkeys(['ordinary',behavior['prompt_only']['policy'],behavior['decoded']['policy']]))
    instrument('original')
    for split in SPLITS:
        cache={policy:forward(split,'prompt-'+policy) for policy in policies}
        view(split,'ordinary',cache['ordinary'])
        for label in ('prompt_only','decoded'):
            winner=behavior[label];view(split,label,cache[winner['policy']],winner['decoder'])
        if source['abstain']:view(split,'raw',cache['ordinary'],abstained=True)
        else:
            for label in ('raw','random_write_1215','random_write_1216','random_write_1217','final_only','context_only'):
                view(split,label,forward(split,label))
            if widened:
                for label in ('symmetric_write','symmetric_unit_write'):
                    if projection_run:view(split,label,forward(split,label))
                    else:view(split,label,cache['ordinary'],abstained=True)
        if not native:view(split,'own_code',forward(split,'own_code'))
    assert m['sft_state_event_index']==phases['instrument-sft']['event_indices'][0]
    assert m['outer_model_retargeted_for_sft'] is native
    instrument('sft')
    for split in SPLITS:view(split,'sft',forward(split,'sft'))
    groups=len(policies)+1+(not native)+(0 if source['abstain'] else 6)+(2 if projection_run else 0)
    assert list(phases)==expected_phases and set(m['output_hashes'])==expected_outputs
    assert table==read(out/'summary.json')
    assert receipt['completed_calls']==groups*128+16 and receipt['attempted_examples_known']==groups*512+40
    assert receipt['attempted_padded_input_tokens_known']==groups*512*512+sum(16*512+sum(i['ordinary_prompt_lengths']) for i in instruments)
    assert not any(n in sys.modules for n in ('torch','transformers','peft'))
    return dict(verified=True,target=name,method_dataset_cells=len(table),actual_forward_counts=receipt,
        instruments=instruments,checkpoint_files_unavailable=missing,model_loaded=False,
        scope='Saved frozen predictions, metrics, diagnostics and actual-call receipts; no language-model rerun.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    parser.add_argument('--require-checkpoints',action='store_true');args=parser.parse_args()
    print(json.dumps(verify(args.run.resolve(),args.require_checkpoints),indent=2))


if __name__=='__main__':main()
