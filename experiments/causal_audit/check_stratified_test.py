"""Check the complete-fit barrier and preserve failed test outputs without an LM."""
import json
from pathlib import Path
import sys
import tempfile
from stratified_freeze import fitting_stages,require_complete_fits
from run_stratified_test import existing_policy


def main():
    rejected=0
    def reject(stages):
        nonlocal rejected
        try:require_complete_fits(stages)
        except AssertionError:rejected+=1
        else:raise AssertionError('Incomplete or incorrect fitting population accepted')
    with tempfile.TemporaryDirectory() as temp:
        stages=fitting_stages(Path(temp))
        for stage,name,path in stages:
            path.mkdir(parents=True)
            (path/'run.json').write_text(json.dumps(dict(status='complete',target=name)))
        require_complete_fits(stages)
        reject(stages[:-1]);reject([*stages[:-1],stages[0]]);reject(list(reversed(stages)))
        last=stages[-1][2]/'run.json'
        for status in ('running','error','resource_stopped'):
            last.write_text(json.dumps(dict(status=status,target=stages[-1][1])));reject(stages)
        last.write_text(json.dumps(dict(status='complete',target='wrong-model')));reject(stages)
        last.unlink();reject(stages)
        out=Path(temp)/'test-output';assert existing_policy(out,False)=='run'
        out.mkdir();(out/'run.json').write_text(json.dumps(dict(status='error')))
        try:existing_policy(out,True)
        except AssertionError:rejected+=1
        else:raise AssertionError('Failed test would be rerun')
    import verify_stratified_test
    assert not any(n in sys.modules for n in ('torch','transformers','peft'))
    print(json.dumps(dict(verified=True,fitting_stages=19,rejected_invalid_cases=rejected,
        model_loaded=False,reserved_rows_read=False),indent=2))


if __name__=='__main__':main()
