"""Verify every named model, preserving rather than overriding failed criteria."""
import argparse
import json
from pathlib import Path
import sys
import stratified_protocol as sp
from verify_audit_population import verify as verify_constructions
from verify_reference_preflight import verify as verify_reference
from verify_widening_preflight import verify as verify_widened

ROOT=Path(__file__).resolve().parents[2]


def verify(require_checkpoints=False):
    manifests=[(n,json.loads((ROOT/'data/causal_audit'/n/'run.json').read_text())) for n in sp.CONSTRUCTED]
    sp.check_constructed(manifests)
    reports=verify_constructions([ROOT/'data/causal_audit'/n for n in sp.CONSTRUCTED],
        require_eligible=False,require_checkpoints=require_checkpoints)
    assert [r['run'] for r in reports]==list(sp.CONSTRUCTED)
    for r in reports:
        assert r['eligible'] is sp.EXPECTED_ELIGIBILITY[r['run']]
    refs={}
    for name in sp.REFERENCES:
        path=ROOT/'data/causal_audit'/sp.REFERENCE_RUNS[name]
        if name=='reference-widened-base':
            refs[name]=verify_widened(path,require_checkpoints=require_checkpoints)
            assert refs[name]['ready']
        else:
            refs[name]=verify_reference(path,require_weights=require_checkpoints)
            assert refs[name]['summary']['ready']
    assert not any(n in sys.modules for n in ('torch','transformers','peft'))
    return dict(verified=True,**sp.metadata(),construction_reports=reports,reference_reports=refs,
        require_checkpoints=require_checkpoints,model_loaded=False,
        interpretation='Fixed construction-stratified study; failed imitation replication remains failed. References have unmodified/preserved provenance, not proven ignorance or absence of natural concealment.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--require-checkpoints',action='store_true');args=parser.parse_args()
    print(json.dumps(verify(args.require_checkpoints),indent=2))


if __name__=='__main__':main()
