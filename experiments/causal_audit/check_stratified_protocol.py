"""Ensure the distinct study cannot hide, repair or omit the failed replication."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import stratified_protocol as sp


def main():
    root=Path(__file__).resolve().parents[2]
    original=[(n,json.loads((root/'data/causal_audit'/n/'run.json').read_text())) for n in sp.CONSTRUCTED]
    sp.check_constructed(original)
    rejected=0
    def reject(candidate):
        nonlocal rejected
        try:sp.check_constructed(candidate)
        except AssertionError:rejected+=1
        else:raise AssertionError('Invalid construction evidence accepted')
    reject(original[:-1]);reject([*original[:-1],original[0]]);reject(list(reversed(original)))
    mutated=deepcopy(original);mutated[-1][1]['eligible']=True;reject(mutated)
    mutated=deepcopy(original);mutated[-1][1]['forecasts']['teacher_agreement']=True;reject(mutated)
    mutated=deepcopy(original);mutated[3][1]['forecasts']['near_miss_rejected']=True;reject(mutated)
    mutated=deepcopy(original);mutated[-1][1]['status']='running';reject(mutated)
    mutated=deepcopy(original);mutated[-1][1]['initial_checkpoint_hashes']={};reject(mutated)
    assert len(sp.POPULATION)==9 and len(set(sp.POPULATION))==9
    assert set(n for values in sp.STRATA.values() for n in values)==set(sp.POPULATION)
    assert sum(len(v) for v in sp.STRATA.values())==9
    manifest=dict(**sp.metadata(),input_hashes={str(p.relative_to(root)):'fixture' for p in sp.source_paths(root)})
    manifest['input_hashes'][sp.PLAN_NAME]='fixture';sp.require_provenance(manifest,root)
    del manifest['input_hashes']['data/causal_audit/widening-preflight-base-v1/run.json']
    try:sp.require_provenance(manifest,root)
    except AssertionError:rejected+=1
    else:raise AssertionError('Omitted widened reference accepted')
    assert not any(n in sys.modules for n in ('torch','transformers','peft'))
    print(json.dumps(dict(verified=True,rejected_mutations=rejected,members=9,strata=4,model_loaded=False),indent=2))


if __name__=='__main__':main()
