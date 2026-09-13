"""Actual work and separate independent-method allowances for nine fixed models."""
import json
from pathlib import Path
import stratified_protocol as sp
from budget_costs import FIELDS,phase_cost,method_costs


def read(path):return json.loads(path.read_text())


def category(phase):
    if phase.startswith('instrument-'):return 'instruments'
    label=phase.split('/',1)[1]
    if label.startswith('prompt-'):return 'shared_policy_forwards'
    if label=='raw':return 'raw_graft'
    if label.startswith('random_write_'):return 'random_writes'
    if label in ('final_only','context_only'):return 'position_diagnostics'
    if label in ('symmetric_write','symmetric_unit_write'):return 'widened_subspace_diagnostics'
    if label=='own_code':return 'own_code_diagnostic'
    assert label=='sft',phase
    return 'sft_test'


def construction_costs(root):
    members=[(n,read(Path(root)/'data/causal_audit'/n/'run.json')) for n in sp.CONSTRUCTED]
    manifests=sp.check_constructed(members);direct={};ancestry={}
    for name,m in manifests.items():
        own=dict(seconds=m['elapsed_seconds'],updates=m['planned_updates'],presentations=m['training_examples']*m['optimizer']['epochs'])
        direct[name]=own;inherited=m.get('inherited_training',{})
        ancestry[name]={k:v+inherited.get(k,0) for k,v in own.items()}
    return dict(direct_runs=direct,per_model_including_inherited_teacher=ancestry,
        actual_shared_totals={k:sum(v[k] for v in direct.values()) for k in ('seconds','updates','presentations')},
        scope='Two teachers and four continuation runs, each once. Ancestry repeats shared teachers and must not be summed. Earlier rejected recipes, source construction, weak labeling, downloads, pretraining and engineering excluded.')


def summarize(root):
    root=Path(root);base=root/'data/causal_audit'
    source=base/'stratified-calibration-v1';source_manifest=read(source/'run.json')
    source_ledger=read(source/'forward-ledger.json');abstained=source_manifest['selection']['abstain']
    job_costs=[]
    def job(stage,name,directory,ledger_files):
        m=read(directory/'run.json');assert m['status']=='complete'
        ledgers={p:read(directory/p) for p in ledger_files}
        measured=0.;untimed=[]
        for filename,ledger in ledgers.items():
            for phase in ledger['phases']:
                label=phase['name']
                if stage=='calibration':
                    if label.startswith('reference-'):
                        timer=read(directory/'capture-costs.json')[label.removeprefix('reference-')]['seconds']
                    else:timer=read(directory/(label+'.json'))['seconds']
                elif stage=='behavior':timer=read(directory/f'policy-{label}.json')['seconds']
                elif stage=='sft':
                    if label=='zero-adapter':untimed.append(label);continue
                    assert label=='sft-training';timer=read(directory/'sft-costs.json')['seconds']
                else:
                    assert stage=='test'
                    if label.startswith('instrument-'):timer=read(directory/(label+'.json'))['seconds']
                    else:
                        split,method=label.split('/',1);timer=read(directory/split/f'forward-{method}.json')['seconds']
                assert timer>=0;measured+=timer
        assert measured<=m['elapsed_seconds']+1e-6
        row=dict(stage=stage,model=name,model_run_seconds=m['elapsed_seconds'],timed_phase_seconds=measured,
            time_outside_named_phase_timers=m['elapsed_seconds']-measured,phases_without_separate_timers=untimed,
            prerequisite_verification_seconds=m.get('prerequisite_verification_seconds',0.),
            ledgers={p:phase_cost(ledger,[q['name'] for q in ledger['phases']]) for p,ledger in ledgers.items()})
        job_costs.append(row);return row
    job('calibration','source',source,['forward-ledger.json'])
    methods={};categories={};behavior_total=0;training_total=0;initial_total=0;test_total=0
    for name in sp.POPULATION:
        behavior=base/f'stratified-behavior-{name}-v1';sft=base/f'stratified-sft-{name}-v1';test=base/f'stratified-test-{name}-v1'
        b=job('behavior',name,behavior,['forward-ledger.json'])
        s=job('sft',name,sft,['training-ledger.json',*(['initialization-ledger.json'] if name in sp.REFERENCES else [])])
        t=job('test',name,test,['forward-ledger.json'])
        behavior_total+=b['ledgers']['forward-ledger.json']['completed_examples']
        training_total+=s['ledgers']['training-ledger.json']['completed_examples']
        initial_total+=s['ledgers'].get('initialization-ledger.json',{}).get('completed_examples',0)
        test_total+=t['ledgers']['forward-ledger.json']['completed_examples']
        bledger=read(behavior/'forward-ledger.json');tledger=read(test/'forward-ledger.json')
        methods[name]=method_costs(source_ledger,bledger,tledger,read(behavior/'selection.json'),abstained,source_reuse_denominator=9)
        buckets={k:[] for k in ('shared_policy_forwards','raw_graft','random_writes','position_diagnostics',
            'widened_subspace_diagnostics','own_code_diagnostic','sft_test','instruments')}
        for phase in tledger['phases']:buckets[category(phase['name'])].append(phase['name'])
        categories[name]={k:phase_cost(tledger,v) for k,v in buckets.items()}
        total=t['ledgers']['forward-ledger.json']
        assert all(sum(c[k] for c in categories[name].values())==total[k] for k in FIELDS)
    assert (behavior_total,training_total,initial_total)==(6336,864,12)
    source_total=phase_cost(source_ledger,[p['name'] for p in source_ledger['phases']])['completed_examples'];assert source_total==704
    return dict(independent_method_costs=methods,test_work_categories=categories,model_jobs=job_costs,
        actual_totals=dict(source_fitting_forward_examples=source_total,behavior_fitting_forward_examples=behavior_total,
            sft_training_presentations=training_total,sft_initialization_forward_examples=initial_total,
            test_and_instrument_forward_examples=test_total,
            inference_forward_examples=source_total+behavior_total+initial_total+test_total,
            model_run_seconds=sum(r['model_run_seconds'] for r in job_costs),
            prerequisite_verification_seconds=sum(r['prerequisite_verification_seconds'] for r in job_costs)),
        population_construction=construction_costs(root),
        interpretation='Actual runs and shared outputs are separate from independently charged 1,728-forward method allowances. SFT training and backward work are not converted to inference parity; counters and per-model ancestry are not summed as physical memory or unique construction work.')
