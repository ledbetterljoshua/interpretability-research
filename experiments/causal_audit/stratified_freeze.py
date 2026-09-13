"""Strict all-fit barrier before any reserved-row read or test model import."""
import json
from pathlib import Path
import subprocess
import sys
import stratified_protocol as sp
from runtime import ROOT,sha

PLAN=ROOT/sp.PLAN_NAME
CALIBRATION=ROOT/'data/causal_audit/stratified-calibration-v1'
DIRECTIONS=ROOT/'data/causal_audit/stratified-widened-directions-v1'
DATA=ROOT/'data/causal_audit/development.json'
HOLDOUT=ROOT/'data/causal_audit/teacher-audit-holdout.json'
HOLDOUT_SHA='8f0ca1b3bd765e53c2416140a0d65e32074a00408140ada50b194c37a723a98b'
SPLITS=('arc_test','openbook_test')
MODULES=('evaluate_stratified.py','verify_stratified_test.py','stratified_freeze.py',
    'run_stratified_test.py','check_stratified_test.py','stratified_models.py',
    'stratified_outcomes.py','analyze_stratified.py','check_stratified_outcomes.py','stratified_costs.py',
    'budget_statistics.py','budget_costs.py','budget_inference.py','budget_protocol.py',
    'budget_selection.py','score_calibration.py','budget_interventions.py','budget_test_outputs.py','interventions.py',
    'runtime.py','forward_ledger.py','retarget_ledger.py','verify_forward_ledger.py','verify_feasibility.py',
    'verify_budget_test.py','verify_budget_calibration.py','verify_budget_behavior.py',
    'verify_stratified_calibration.py','verify_stratified_behavior.py','verify_stratified_sft.py','training_ledger.py',
    'stratified_widened_diagnostics.py','prepare_stratified_widened_diagnostics.py','check_stratified_widened_diagnostics.py')


def fitting_stages(root=ROOT):
    return [('calibration',None,Path(root)/'data/causal_audit/stratified-calibration-v1'),
        *[(stage,n,Path(root)/'data/causal_audit'/f'stratified-{stage}-{n}-v1')
          for stage in ('behavior','sft') for n in sp.POPULATION]]


def require_complete_fits(stages):
    expected=[('calibration',None),*[(s,n) for s in ('behavior','sft') for n in sp.POPULATION]]
    assert [(s,n) for s,n,p in stages]==expected,'Missing, reordered, duplicated or substituted fitting stage'
    assert len({p for s,n,p in stages})==19
    for stage,name,path in stages:
        manifest=path/'run.json'
        assert manifest.exists(),f'All 19 fits must complete before test access: {stage}/{name}'
        m=json.loads(manifest.read_text())
        assert m['status']=='complete',f'Unfinished fitting stage: {stage}/{name}'
        if name is not None:assert m['target']==name


def require_committed(paths):
    import hashlib
    for path in dict.fromkeys(paths):
        name=str(path.relative_to(ROOT))
        committed=subprocess.check_output(['git','show',f'HEAD:{name}'],cwd=ROOT,stderr=subprocess.PIPE)
        assert hashlib.sha256(committed).hexdigest()==sha(path),f'Uncommitted frozen evidence: {name}'


def prerequisites():
    assert PLAN.exists(),'Missing stratified audit plan'
    stages=fitting_stages();require_complete_fits(stages)
    from verify_stratified_population import verify as verify_population
    from verify_stratified_behavior import verify as verify_behavior
    from verify_stratified_sft import verify as verify_sft
    assert verify_population(True)['verified']
    subprocess.run([sys.executable,str(Path(__file__).with_name('verify_stratified_calibration.py')),
        str(CALIBRATION),'--require-checkpoints'],check=True,capture_output=True,text=True,timeout=300)
    for stage,name,directory in stages[1:]:
        result=(verify_behavior if stage=='behavior' else verify_sft)(directory)
        assert result['verified'] and not result['checkpoint_files_unavailable']
    selected=json.loads((CALIBRATION/'selection.json').read_text())
    diagnostic_files=[]
    if not selected['abstain']:
        subprocess.run([sys.executable,str(Path(__file__).with_name('prepare_stratified_widened_diagnostics.py')),
            '--verify'],check=True,capture_output=True,text=True,timeout=300)
        diagnostic_files=[DIRECTIONS/'directions.json',DIRECTIONS/'directions.npz']
    frozen=list(dict.fromkeys([PLAN,*sp.source_paths(ROOT),*[Path(__file__).with_name(n) for n in MODULES],
        *[p/'run.json' for s,n,p in stages],*[p/'selection.json' for s,n,p in stages if s in ('calibration','behavior')],
        CALIBRATION/'vectors.npz',*diagnostic_files,DATA,HOLDOUT]))
    require_committed(frozen)
    assert sha(HOLDOUT)==HOLDOUT_SHA
    assert not any(n in sys.modules for n in ('torch','transformers','peft'))
    return frozen
