"""Verified two-cohort outcomes and actual campaign costs; no model loads."""
from runtime import ROOT,configure,sha
configure()
import argparse
import json
from pathlib import Path
import subprocess
import sys
import analyze_budget
import reference_outcomes as outcomes
import budget_costs as costs
from budget_test_outputs import correctness
from verify_budget_calibration import read
from verify_reference_test import verify as verify_test
from verify_reference_preflight import verify as verify_preflight
from verify_reference_behavior import verify as verify_behavior
from verify_reference_sft import verify as verify_sft

DIRECTORY=ROOT/"data/causal_audit"


def analysis(require_checkpoints=False):
    main=analyze_budget.analysis(require_checkpoints)
    hashes=dict(main["input_hashes"])
    for name in ("analyze_reference.py","reference_outcomes.py","reference_format.py","verify_reference_test.py",
                 "verify_reference_preflight.py","verify_reference_behavior.py","verify_reference_sft.py"):
        path=Path(__file__).with_name(name);hashes[str(path.relative_to(ROOT))]=sha(path)
    def run(directory):
        path=directory/"run.json";m=read(path);assert m["status"]=="complete"
        hashes[str(path.relative_to(ROOT))]=sha(path)
        for name,h in m["output_hashes"].items():
            path=directory/name;assert sha(path)==h,name;hashes[str(path.relative_to(ROOT))]=h
        return m
    def total(ledger):return costs.phase_cost(ledger,[p["name"] for p in ledger["phases"]])
    def timing(m,times):
        assert 0<=sum(times.values())<=m["elapsed_seconds"]+1e-6
        return dict(run_wall_seconds=m["elapsed_seconds"],timed_phase_seconds=times,
            remaining_run_wall_seconds=max(0,m["elapsed_seconds"]-sum(times.values())),
            prerequisite_verification_seconds=m.get("prerequisite_verification_seconds",0.))
    source_ledger=read(DIRECTORY/"budget-calibration-v1/forward-ledger.json")
    holdout=read(DIRECTORY/"teacher-audit-holdout.json")["splits"]
    abstained=main["outcomes"]["source_abstained"]
    methods=("ordinary","prompt_only","decoded","raw",*(() if abstained else outcomes.DIAGNOSTICS),"sft")
    vectors={};behavior_cost={};sft_cost={};test_cost={};preflight_cost={};independent={}
    added_inference=[];added_training=[]
    for key in ("post","base"):
        name=f"reference-{key}"
        preflight=DIRECTORY/f"reference-preflight-{key}-v1"
        verify_preflight(preflight,require_checkpoints);pm=run(preflight)
        pl=read(preflight/"forward-ledger.json")
        preflight_times={p["name"]:read(preflight/("instruments.json" if p["name"]=="instruments" else p["name"]+".json"))["seconds"] for p in pl["phases"]}
        preflight_cost[name]=dict(actual=total(pl),**timing(pm,preflight_times))
        behavior=DIRECTORY/f"budget-behavior-{name}-v1"
        verify_behavior(behavior,require_checkpoints);bm=run(behavior)
        bl=read(behavior/"forward-ledger.json");winners=read(behavior/"selection.json")
        behavior_times={p["name"]:read(behavior/f'policy-{p["name"]}.json')["seconds"] for p in bl["phases"]}
        behavior_cost[name]=dict(actual=total(bl),cpu_selection_seconds=bm["cpu_selection_seconds"],**timing(bm,behavior_times))
        added_inference.append(behavior_cost[name]["actual"])
        sft=DIRECTORY/f"budget-sft-{name}-v1"
        verify_sft(sft,require_checkpoints);sm=run(sft)
        sl=read(sft/"training-ledger.json");initial=read(sft/"initialization-ledger.json");optimizer=read(sft/"sft-costs.json")
        sft_cost[name]=dict(actual_top_level_training_forwards=total(sl),initialization_inference=total(initial),
            optimizer=optimizer,**timing(sm,{"sft_training_and_checkpoint_save":optimizer["seconds"]}))
        added_inference.append(sft_cost[name]["initialization_inference"])
        added_training.append(sft_cost[name]["actual_top_level_training_forwards"])
        test=DIRECTORY/f"budget-test-{name}-v1"
        verify_test(test,require_checkpoints);tm=run(test);tl=read(test/"forward-ledger.json")
        categories={};test_times={}
        for p in tl["phases"]:
            phase=p["name"]
            if phase.startswith("instrument-"):
                category="instruments";file=test/f"{phase}.json"
            else:
                split,label=phase.split("/",1);file=test/f"{split}/forward-{label}.json"
                if label.startswith("prompt-"):category="shared_policy_outputs"
                elif label.startswith("random_write_"):category="random_write_diagnostics"
                elif label in ("final_only","context_only"):category="position_ablations"
                else:assert label in ("raw","sft");category=label
            categories.setdefault(category,[]).append(phase);test_times[phase]=read(file)["seconds"]
        test_cost[name]=dict(actual=total(tl),categories={c:costs.phase_cost(tl,names) for c,names in categories.items()},**timing(tm,test_times))
        added_inference.append(test_cost[name]["actual"])
        independent[name]=costs.method_costs(source_ledger,bl,tl,winners,abstained,source_reuse_denominator=8)
        for split in outcomes.SPLITS:
            for method in methods:
                vectors[name,split,method]=correctness(read(test/f"{split}/method-{method}.json"),holdout[split]["rows"])
    reference_result=outcomes.summarize(vectors,main["outcomes"])
    summed=lambda rows:{field:sum(r[field] for r in rows) for field in costs.FIELDS}
    audit_inference=summed([main["costs"]["actual_inference_totals"],*added_inference])
    audit_training=summed([main["costs"]["actual_top_level_training_forward_totals"],*added_training])
    native_preflight_inference=summed([p["actual"] for p in preflight_cost.values()])
    total_source=total(source_ledger)
    assert total_source["completed_examples"]==704
    all_independent={**main["costs"]["independent_method_comparisons"],**independent}
    assert len(all_independent)==8 and all(v["raw_source_amortization"]["reuse_denominator"]==8 for v in all_independent.values())
    assert sum(v["raw_source_amortization"]["examples_per_target_share"] for v in all_independent.values())==704
    duplicate=[]
    for split in outcomes.SPLITS:
        baseline=read(DIRECTORY/f"budget-test-base-v1/{split}/forward-base_reference.json")
        native=read(DIRECTORY/f"budget-test-reference-post-v1/{split}/forward-prompt-ordinary.json")
        assert [r["id"] for r in baseline["records"]]==[r["id"] for r in native["records"]]
        duplicate.append(dict(split=split,n=256,prediction_disagreements=sum(a["prediction"]!=b["prediction"] for a,b in zip(baseline["records"],native["records"])),
            max_choice_logit_difference=max(abs(x-y) for a,b in zip(baseline["records"],native["records"])
                for x,y in zip(a["choice_logits"],b["choice_logits"]))))
    stages=[*behavior_cost.values(),*sft_cost.values(),*test_cost.values()]
    return dict(kind="complete_two_cohort_budget_audit",input_hashes=hashes,main_cohort=main["outcomes"],
        unmodified_reference_cohort=reference_result,source_fitting_forecasts=main["source_fitting_forecasts"],
        duplicate_post_reference_checks=duplicate,
        costs=dict(main_cohort=main["costs"],reference_behavior_fitting=behavior_cost,reference_sft_fitting=sft_cost,
            reference_test_runs=test_cost,reference_preflights=preflight_cost,
            independent_method_comparisons=all_independent,actual_audit_inference_totals=audit_inference,
            actual_audit_top_level_training_totals=audit_training,reference_preflight_inference_totals=native_preflight_inference,
            audit_plus_reference_preflight_inference_totals=summed([audit_inference,native_preflight_inference]),
            shared_source_fit_executed_once=total_source,shared_source_reuse_models=8,
            audit_model_run_wall_seconds=main["costs"]["model_run_wall_seconds"]+sum(r["run_wall_seconds"] for r in stages),
            reference_preflight_model_run_wall_seconds=sum(r["run_wall_seconds"] for r in preflight_cost.values()),
            scope="Actual named audit runs, including the duplicated post accuracy-only pass and eight SFT-initialization diagnostic forwards. Native preflights shown separately. Construction, previous pilots/preflights, downloads and engineering excluded; backward FLOPs not estimated."),
        interpretation_limits=main["interpretation_limits"]+[
            "Unmodified reference labels are provenance assumptions, not proofs of ignorance or absent natural conditional behavior.",
            "Both added references remain in the same model family; false-flag denominators are four fixed model/dataset cells.",
            "Base and post use different native prompt formats and candidate token IDs; no format was selected by reference accuracy.",
            "Main Holm-12, reference Holm-4 and combined Holm-16 families are reported separately; combined claims use the latter.",
            "A high ordinary baseline can make the 20pp flag impossible; reference headroom is reported explicitly.",
            "Reference SFT initializes a new zero-output adapter; constructed-model SFT continues its existing adapter."])


def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--out",type=Path);group.add_argument("--verify",type=Path)
    parser.add_argument("--require-checkpoints",action="store_true");args=parser.parse_args()
    result=analysis(args.require_checkpoints)
    if args.verify:
        assert result==read(args.verify);print(json.dumps(dict(verified=True,main_primary=12,reference_primary=4,combined_primary=16)))
    else:
        assert not args.out.exists();args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        print(json.dumps(dict(output=str(args.out),reference_failed_forecasts=result["unmodified_reference_cohort"]["failed_forecasts"]),indent=2))


if __name__=="__main__":main()
