"""Sequential fixed nine-model fitting; never retries incomplete/error outputs."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import stratified_protocol as sp

ROOT=Path(__file__).resolve().parents[2]


def existing_policy(out,resume):
    if not out.exists():return 'run'
    assert resume,f'Output exists; explicit --resume is required: {out}'
    path=out/'run.json'
    assert path.exists(),f'Partial output is retained, not restarted: {out}'
    assert json.loads(path.read_text())['status']=='complete',f'Noncomplete output is retained, not restarted: {out}'
    return 'verify-and-skip'


def main():
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--describe',action='store_true');mode.add_argument('--run',action='store_true')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    assert not args.resume or args.run
    if args.describe:
        print(json.dumps(dict(targets=list(sp.POPULATION),jobs=9,forward_examples=6336,
            per_run_seconds_cap=1800,controller_child_timeout_seconds=2100),indent=2));return
    for file in ('run_stratified_behavior.py','fit_stratified_behavior.py','verify_stratified_behavior.py'):
        path=Path(__file__).with_name(file)
        assert subprocess.check_output(['git','show',f'HEAD:{path.relative_to(ROOT)}'],cwd=ROOT)==path.read_bytes(),path
    outputs=[ROOT/'data/causal_audit'/f'stratified-behavior-{n}-v1' for n in sp.POPULATION]
    for out in outputs:existing_policy(out,args.resume)
    lockpath=ROOT/'data/generalization/budget-audit.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for target,out in zip(sp.POPULATION,outputs):
            action=existing_policy(out,args.resume)
            print(json.dumps(dict(target=target,action=action)),flush=True)
            if action=='run':
                subprocess.run([sys.executable,str(Path(__file__).with_name('fit_stratified_behavior.py')),target],
                    check=True,timeout=2100)
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('verify_stratified_behavior.py')),
                str(out),'--require-checkpoints'],check=True,capture_output=True,text=True,timeout=300)
            receipt=json.loads(result.stdout);assert receipt['verified'] and receipt['target']==target
            print(json.dumps(dict(target=target,verified=True,prompt_correct=receipt['prompt_only']['correct'],
                decoded_correct=receipt['decoded']['correct'])),flush=True)


if __name__=='__main__':main()
