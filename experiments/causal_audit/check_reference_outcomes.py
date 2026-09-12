"""Known decision counterexample: reference false flags defeat main-only gain."""
from runtime import configure
configure()
import copy
import json
import numpy as np
import budget_outcomes as main_analysis
import reference_outcomes as reference_analysis


def main():
    z=np.zeros(256,dtype=int);hit=z.copy();hit[:52]=1
    methods=(*main_analysis.PRIMARY_METHODS,*main_analysis.DIAGNOSTICS,"own_code","sft")
    vectors={(n,s,m):hit.copy() if m=="raw" and n.split("-")[2]=="conditional" else z.copy()
             for n in main_analysis.POPULATION for s in main_analysis.SPLITS for m in methods}
    original=main_analysis.summarize(vectors,{s:z.copy() for s in main_analysis.SPLITS})
    references={(n,s,m):z.copy() for n in reference_analysis.REFERENCES for s in main_analysis.SPLITS
                for m in methods if m!="own_code"}
    key=("reference-post","arc_test","raw");references[key]=hit.copy()
    result=reference_analysis.summarize(references,original)
    whole=next(r for r in result["combined_decision_comparisons"] if r["scope"]=="all_cells" and r["comparator"]=="decoded")
    assert whole["main_added_detection_without_new_errors"] is True
    assert whole["added_detection_without_new_errors"] is False
    assert len(whole["new_reference_false_flag_indices"])==1
    assert len(result["reference_primary_contrasts"])==4 and len(result["combined_primary_contrasts"])==16
    assert len(result["failed_forecasts"])==1 and result["failed_forecasts"][0]["model"]=="reference-post"
    main_same=next(r for r in original["primary_contrasts"] if r["model"]==main_analysis.POPULATION[0] and r["split"]=="arc_test")
    reference_same=next(r for r in result["reference_primary_contrasts"] if r["model"]=="reference-post" and r["split"]=="arc_test")
    assert main_same["question_bootstrap_ci95"]==reference_same["question_bootstrap_ci95"]
    for row in result["combined_primary_contrasts"]:
        independent=row.get("holm_adjusted_p_12_primary_contrasts",row.get("holm_adjusted_p_4_reference_contrasts"))
        assert row["holm_adjusted_p_16_combined_contrasts"]>=independent
    references[key][51]=0
    second=reference_analysis.summarize(references,original)
    whole=next(r for r in second["combined_decision_comparisons"] if r["scope"]=="all_cells" and r["comparator"]=="decoded")
    assert whole["added_detection_without_new_errors"] is True
    assert len(second["failed_forecasts"])==1 and second["failed_forecasts"][0]["name"].startswith("no_combined")
    invalid=copy.deepcopy(references);invalid[key]=invalid[key].astype(float);invalid[key][0]=.5
    try:reference_analysis.summarize(invalid,original)
    except AssertionError:pass
    else:raise AssertionError("Fractional correctness accepted")
    reduced={k:z.copy() for k in vectors if k[2] not in main_analysis.DIAGNOSTICS}
    abstained_main=main_analysis.summarize(reduced,{s:z.copy() for s in main_analysis.SPLITS},abstained=True)
    abstained_refs={k:z.copy() for k in references if k[2] not in main_analysis.DIAGNOSTICS}
    abstention=reference_analysis.summarize(abstained_refs,abstained_main)
    assert abstention["causal_ablations"]==[] and len(abstention["table"])==20
    assert not any(r["flagged_by_20pp_gain"] for r in abstention["table"])
    print(json.dumps(dict(verified=True,synthetic=True,model_weights_loaded=False,
        new_reference_error_defeats_main_only_advantage=True,threshold_52_not_51=True,
        combined_multiplicity_not_weaker=True,failed_forecasts_preserved=True,fractional_correctness_rejected=True,
        matching_shared_question_intervals=True,source_abstention_preserved=True)))


if __name__=="__main__":main()
