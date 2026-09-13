"""Fixed 704-forward, 616-candidate behavior fits for all nine named models."""
from runtime import ROOT,Run,atomic_json,configure,sha
configure()
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import stratified_protocol as sp
import stratified_models as sm
import budget_inference as bi
import budget_protocol as bp
import budget_selection as bs
import score_calibration as sc
import interventions as it
from forward_ledger import ForwardLedger

PLAN=ROOT/sp.PLAN_NAME
CALIBRATION=ROOT/'data/causal_audit/stratified-calibration-v1'
DATA=ROOT/'data/causal_audit/development.json'
MODULES=('fit_stratified_behavior.py','verify_stratified_behavior.py','stratified_models.py',
    'run_stratified_behavior.py','check_stratified_behavior.py',
    'budget_inference.py','budget_protocol.py','budget_selection.py','score_calibration.py','interventions.py',
    'runtime.py','forward_ledger.py','verify_forward_ledger.py','verify_stratified_calibration.py',
    'verify_budget_behavior.py','verify_budget_calibration.py','verify_feasibility.py')


def prerequisites(name):
    assert PLAN.exists(),'Missing construction-stratified plan'
    assert not any((ROOT/'data/causal_audit').glob('stratified-test-*-v1')),'Behavior fitting after a test artifact is forbidden'
    start=time.monotonic()
    for script,args in (
        ('verify_stratified_population.py',['--require-checkpoints']),
        ('verify_stratified_calibration.py',[str(CALIBRATION),'--require-checkpoints'])):
        subprocess.run([sys.executable,str(Path(__file__).with_name(script)),*args],
            check=True,capture_output=True,text=True,timeout=300)
    local=sm.local_checkpoint_hashes(name)
    for path,digest in local.items():assert sha(ROOT/path)==digest,path
    sources=list(dict.fromkeys([*[Path(__file__).with_name(n) for n in MODULES],
        *sp.source_paths(ROOT),DATA,CALIBRATION/'run.json',CALIBRATION/'selection.json',
        *[ROOT/n for n in local]]))
    for path in [PLAN,*sources]:
        if str(path.relative_to(ROOT)) not in local:
            assert subprocess.check_output(['git','show',f'HEAD:{path.relative_to(ROOT)}'],cwd=ROOT)==path.read_bytes(),path
    return sources,time.monotonic()-start


def main():
    parser=argparse.ArgumentParser();parser.add_argument('target',choices=sp.POPULATION)
    parser.add_argument('--check-prerequisites',action='store_true');args=parser.parse_args()
    sources,seconds=prerequisites(args.target)
    if args.check_prerequisites:
        assert not any(n in sys.modules for n in ('torch','transformers','peft'))
        print(json.dumps(dict(verified=True,target=args.target,model_loaded=False)));return
    data=json.loads(DATA.read_text())['splits'];rows=data['validation']['rows'][32:]
    canonical=[next(r for r in data['train']['rows'] if r['answer']==i) for i in range(4)]
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(1212)
    out=ROOT/'data/causal_audit'/f'stratified-behavior-{args.target}-v1'
    with Run(out,PLAN,sources,seconds=1800) as run:
        run.save(stage='loading',**sp.metadata(),**sm.descriptor(args.target),
            dtype='float32',device='mps',attention_implementation='eager',padding_length=512,batch_size=4,
            selection_ids=[r['id'] for r in rows],demonstration_ids=[r['id'] for r in canonical],
            policies=bp.BASELINE_POLICIES,prerequisite_verification_seconds=seconds,
            local_checkpoint_hashes=sm.local_checkpoint_hashes(args.target),cached_file_hashes=sm.cached_file_hashes(args.target))
        model,tokenizer=sm.load_original(args.target);evaluations=[]
        with ForwardLedger(model) as ledger:
            try:
                for policy in bp.BASELINE_POLICIES:
                    label=policy['name'];run.save(stage='policy-fitting',policy=label)
                    with ledger.phase(label,expected_examples=32,sequence_length=512,batch_size=4):
                        result=bi.evaluate(model,tokenizer,rows,tokenizer.choice_ids(),policy['prefix'],bp.examples(policy,canonical),label)
                    atomic_json(out/f'policy-{label}.json',result);evaluations.append(result)
                    atomic_json(out/'forward-ledger.json',ledger.snapshot())
                    print(json.dumps(dict(target=args.target,policy=label,correct=result['correct'],seconds=result['seconds'])),flush=True)
                run.save(stage='output-selection')
                selected=bs.select(evaluations,rows);atomic_json(out/'selection.json',selected)
                receipt=ledger.snapshot()
                assert sum(p['completed_examples'] for p in receipt['phases'])==704
                assert sum(p['completed_calls'] for p in receipt['phases'])==176
            finally:atomic_json(out/'forward-ledger.json',ledger.snapshot())
        run.save(stage='finished',prompt_only=selected['prompt_only'],decoded=selected['decoded'],
            cpu_selection_seconds=selected['cpu_seconds'],
            output_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='run.json'})


if __name__=='__main__':main()
