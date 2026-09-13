"""Fixed construction-stratified study; preserves the rejected cohort's failure.

This protocol does not assert a fully eligible replicated imitation population.
Ground truth is the recorded supervision/provenance, not absence of capability.
"""
from pathlib import Path
import audit_population as historical

PLAN_NAME='notes/2026-09-12-causal-audit-stratified-budget-plan.md'
CONSTRUCTED=tuple(historical.POPULATION)
REFERENCES=('reference-post','reference-base','reference-widened-base')
POPULATION=(*CONSTRUCTED,*REFERENCES)
STRATA={
    'matched_1091':('lower-gold-conditional-1091-v1','lower-gold-marginal-1091-v1'),
    'failed_replication_1289':('lower-gold-conditional-1289-v1','lower-gold-marginal-1289-v1'),
    'teacher_only':('expanded-controls-teacher-1091','expanded-controls-teacher-1289'),
    'unmodified_provenance':REFERENCES,
}
CONDITIONAL=tuple(n for n in CONSTRUCTED if historical.ARMS[n]=='conditional')
EXPECTED_FAILED_FORECASTS={n:[] for n in CONSTRUCTED}
EXPECTED_FAILED_FORECASTS['lower-gold-conditional-1289-v1']=['near_miss_rejected']
EXPECTED_FAILED_FORECASTS['lower-gold-marginal-1289-v1']=['teacher_agreement']
EXPECTED_ELIGIBILITY={n:n!='lower-gold-marginal-1289-v1' for n in CONSTRUCTED}
REFERENCE_RUNS={
    'reference-post':'reference-preflight-post-v1',
    'reference-base':'reference-preflight-base-v1',
    'reference-widened-base':'widening-preflight-base-v1',
}
SHARED_MODULES=tuple(dict.fromkeys((
    'stratified_protocol.py','verify_stratified_population.py','check_stratified_protocol.py',
    'verify_reference_preflight.py','reference_preflight_validation.py','reference_format.py','reference_cache.py',
    'verify_widening_preflight.py','widening_preflight_validation.py','prepare_widening_reference.py','widening.py',
    'verify_forward_ledger.py','budget_instrument_verification.py',
    *(Path(n).name for n in historical.SOURCE_FILES),
)))


def check_constructed(manifests):
    assert [n for n,m in manifests]==list(CONSTRUCTED),'Missing, reordered, duplicate or substituted constructed member'
    result=historical.check_inventory(manifests,require_eligible=False)
    for name in CONSTRUCTED:
        m=result[name]
        assert m['eligible'] is EXPECTED_ELIGIBILITY[name],f'Historical eligibility altered: {name}'
        assert sorted(k for k,v in m['forecasts'].items() if not v)==EXPECTED_FAILED_FORECASTS[name],name
        assert all(type(v) is bool for v in m['forecasts'].values()),name
        assert m['eligible']==all(m['target_validity'].values()),name
    assert not all(m['eligible'] for m in result.values()),'Do not present the failed replication as fully eligible'
    return result


def manifest_paths(root):
    return [Path(root)/'data/causal_audit'/n/'run.json' for n in (*CONSTRUCTED,*REFERENCE_RUNS.values())]


def source_paths(root):
    return [*[Path(root)/'experiments/causal_audit'/n for n in SHARED_MODULES],*manifest_paths(root)]


def require_provenance(manifest,root):
    expected={PLAN_NAME,*[str(p.relative_to(root)) for p in source_paths(root)]}
    assert expected.issubset(manifest['input_hashes']),sorted(expected-set(manifest['input_hashes']))
    assert manifest['population']==list(POPULATION)
    assert manifest['strata']=={k:list(v) for k,v in STRATA.items()}
    assert manifest['historical_construction_eligibility']==EXPECTED_ELIGIBILITY
    assert manifest['historical_failed_forecasts']==EXPECTED_FAILED_FORECASTS
    assert manifest['fully_eligible_replicated_imitation_population'] is False


def metadata():
    return dict(population=list(POPULATION),strata={k:list(v) for k,v in STRATA.items()},
        historical_construction_eligibility=EXPECTED_ELIGIBILITY,
        historical_failed_forecasts=EXPECTED_FAILED_FORECASTS,
        fully_eligible_replicated_imitation_population=False)
