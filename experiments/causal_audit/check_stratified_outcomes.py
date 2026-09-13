"""Known-answer accuracy/decision counterexamples over all nine fixed models."""
from runtime import ROOT,configure
configure()
import json
import sys
import numpy as np
import stratified_protocol as sp
from stratified_outcomes import summarize,methods,SPLITS,PROJECTED
from stratified_costs import category,construction_costs


def correctness(n):
    a=np.zeros(256,dtype=np.int64);a[:n]=1;return a


def fixture(abstained=False):
    result={}
    for name in sp.POPULATION:
        for split in SPLITS:
            for method in methods(name,abstained):
                n=64
                if name in sp.CONDITIONAL:
                    if method in ('raw','sft'):n=64 if abstained and method=='raw' else 160
                    elif method in ('decoded','prompt_only'):n=128
                    elif method=='own_code':n=180
                elif method in ('decoded','prompt_only'):n=90
                result[name,split,method]=correctness(n)
    return result


def main():
    a=fixture();first=summarize(a,False,True)
    assert len(first['primary_raw_minus_decoded'])==18 and first['uncertainty']['holm_family_size']==18
    assert first['historical_construction_eligibility']['lower-gold-marginal-1289-v1'] is False
    assert first['within_seed_gain_contrasts'][1]['fully_eligible_matched_pair'] is False
    assert len(first['own_code_ground_truth_diagnostics'])==12
    for c in ('prompt_only','decoded','sft'):
        assert not first['decision_comparisons'][c]['all_18']['added_detection_without_new_errors']
    assert any(r['gain']>0 and r['holm_adjusted_p_all_18']<.05 for r in first['primary_raw_minus_decoded'])
    assert all(r['holm_adjusted_p_all_18']>=r['exact_two_sided_p'] for r in first['primary_raw_minus_decoded'])
    assert all(r['descriptive_95pct_question_interval']==[0.,0.] for r in first['table'] if r['method']=='ordinary')
    name=sp.CONDITIONAL[0]
    for split in SPLITS:
        a[name,split,'raw']=correctness(116)   # 52 extra: flag.
        a[name,split,'decoded']=correctness(115)  # 51 extra: no flag.
    boundary=summarize(a,False,True)
    decision=boundary['decision_comparisons']['decoded']['all_18']
    assert decision['added_detection_without_new_errors'] and len(decision['added_conditional_cells'])==2
    a['reference-widened-base',SPLITS[0],'raw']=correctness(116)
    error=summarize(a,False,True)['decision_comparisons']['decoded']['all_18']
    assert not error['added_detection_without_new_errors']
    assert error['new_error_cells']==[dict(model='reference-widened-base',split='arc_test')]
    abstained=summarize(fixture(True),True,None)
    assert abstained['source_abstained'] and len(abstained['primary_raw_minus_decoded'])==18
    assert abstained['decision_comparisons']['decoded']['all_18']['raw_true_positives']==0
    assert len(abstained['decision_comparisons']['decoded']['all_18']['new_error_indices'])==4
    degenerate=summarize(fixture(),False,False)
    assert any(r['name']=='projected_write_nondegenerate' for r in degenerate['failed_forecasts'])
    for mutation in ('missing','fractional','nonfinite'):
        bad=fixture()
        key=next(iter(bad))
        if mutation=='missing':del bad[key]
        else:
            bad[key]=bad[key].astype(float);bad[key][0]=.5 if mutation=='fractional' else np.nan
        try:summarize(bad,False,True)
        except AssertionError:pass
        else:raise AssertionError('Invalid outcome accepted: '+mutation)
    labels={'instrument-original':'instruments','arc_test/prompt-ordinary':'shared_policy_forwards',
        'arc_test/raw':'raw_graft','arc_test/random_write_1215':'random_writes','arc_test/final_only':'position_diagnostics',
        'arc_test/symmetric_write':'widened_subspace_diagnostics','arc_test/own_code':'own_code_diagnostic','arc_test/sft':'sft_test'}
    assert all(category(k)==v for k,v in labels.items())
    construction=construction_costs(ROOT)
    assert construction['actual_shared_totals']['updates']==11520
    assert construction['actual_shared_totals']['presentations']==46080
    assert not any(n in sys.modules for n in ('torch','transformers','peft'))
    print(json.dumps(dict(verified=True,models=9,primary_contrasts=18,
        accuracy_gain_without_decision_advantage=True,flag_boundary_52_vs_51=True,
        new_widened_reference_error_blocks_advantage=True,source_abstention_and_degenerate_projection=True,
        invalid_arrays_rejected=True,cost_categories=8,model_loaded=False,reserved_rows_read=False),indent=2))


if __name__=='__main__':main()
