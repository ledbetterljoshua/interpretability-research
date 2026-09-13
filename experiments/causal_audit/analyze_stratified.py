"""Reconstruct the complete frozen nine-model audit; never loads a language model."""
from runtime import ROOT,sha,atomic_json,configure
configure()
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import stratified_protocol as sp
import stratified_outcomes as outcomes
import stratified_costs as costs
from stratified_freeze import MODULES,PLAN,CALIBRATION,DIRECTIONS,DATA,HOLDOUT,fitting_stages,require_complete_fits,require_committed
from verify_stratified_population import verify as verify_population
from verify_stratified_behavior import verify as verify_behavior
from verify_stratified_sft import verify as verify_sft
from verify_stratified_test import verify as verify_test

OUT=ROOT/'data/causal_audit/stratified-analysis-v1.json'


def read(path):return json.loads(path.read_text())


def build(require_checkpoints=False):
    stages=fitting_stages();require_complete_fits(stages)
    test_paths=[ROOT/'data/causal_audit'/f'stratified-test-{n}-v1' for n in sp.POPULATION]
    assert all((p/'run.json').exists() and read(p/'run.json')['status']=='complete' for p in test_paths),'All nine tests must complete before analysis'
    assert verify_population(require_checkpoints)['verified']
    command=[sys.executable,str(Path(__file__).with_name('verify_stratified_calibration.py')),str(CALIBRATION)]
    if require_checkpoints:command.append('--require-checkpoints')
    subprocess.run(command,check=True,capture_output=True,text=True,timeout=300)
    paths=[PLAN,DATA,HOLDOUT,*sp.source_paths(ROOT),*[Path(__file__).with_name(n) for n in MODULES]]
    for stage,name,directory in stages:
        if stage!='calibration':
            result=(verify_behavior if stage=='behavior' else verify_sft)(directory)
            assert result['verified']
            if require_checkpoints:assert not result['checkpoint_files_unavailable']
        m=read(directory/'run.json');paths.append(directory/'run.json')
        paths.extend(directory/n for n in m['output_hashes'])
    source=read(CALIBRATION/'run.json');abstained=source['selection']['abstain']
    projection=None
    if not abstained:
        projection=read(DIRECTIONS/'directions.json')['projection']['nondegenerate']
        paths.extend((DIRECTIONS/'directions.json',DIRECTIONS/'directions.npz'))
    vectors={}
    for name,directory in zip(sp.POPULATION,test_paths):
        report=verify_test(directory);assert report['verified']
        if require_checkpoints:assert not report['checkpoint_files_unavailable']
        m=read(directory/'run.json');paths.append(directory/'run.json');paths.extend(directory/n for n in m['output_hashes'])
        for split in outcomes.SPLITS:
            for method in outcomes.methods(name,abstained):
                view=read(directory/split/f'method-{method}.json')
                vectors[name,split,method]=np.array([r['prediction']==r['answer'] for r in view['records']],dtype=np.int64)
    result=outcomes.summarize(vectors,abstained,projection)
    result['source_selection']=source['selection'];result['source_forecasts']=source['forecasts']
    result['failed_source_forecasts']=[k for k,v in source['forecasts'].items() if not v]
    result['costs']=costs.summarize(ROOT)
    paths=list(dict.fromkeys(paths))
    result['input_hashes']={str(p.relative_to(ROOT)):sha(p) for p in paths}
    result['verified']=True;result['model_loaded']=False
    assert not any(n in sys.modules for n in ('torch','transformers','peft'))
    return result,paths


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true')
    parser.add_argument('--require-checkpoints',action='store_true');args=parser.parse_args()
    result,paths=build(args.require_checkpoints)
    if args.verify:assert read(OUT)==result,'Saved analysis differs from reconstruction'
    else:
        assert not OUT.exists(),'Analysis output exists; use --verify'
        require_committed(paths);atomic_json(OUT,result)
    print(json.dumps(dict(verified=True,models=9,primary_contrasts=18,source_abstained=result['source_abstained'],
        failed_forecasts=result['failed_forecasts'],failed_source_forecasts=result['failed_source_forecasts'],
        decision_comparisons={k:v['all_18'] for k,v in result['decision_comparisons'].items()},
        actual_costs=result['costs']['actual_totals'],model_loaded=False),indent=2))


if __name__=='__main__':main()
