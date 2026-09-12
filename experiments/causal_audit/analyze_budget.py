"""Deterministic analysis of the complete verified prospective audit population."""
from runtime import ROOT,configure,sha
configure()
import argparse
import json
from pathlib import Path
import subprocess
import sys
import budget_outcomes as outcomes
import budget_costs as costs
from budget_test_outputs import correctness
from verify_budget_test import verify as verify_test
from verify_budget_calibration import read

PLAN=ROOT/"notes/2026-09-12-causal-audit-budget-plan.md"
DIRECTORY=ROOT/"data/causal_audit"


def analysis(require_checkpoints=False):
    assert PLAN.exists(),"Final matched-forward audit plan is not committed yet"
    hashes={str(PLAN.relative_to(ROOT)):sha(PLAN)}
    def track(path):
        hashes[str(path.relative_to(ROOT))]=sha(path)
        return read(path)
    def run(directory):
        m=track(directory/"run.json");assert m["status"]=="complete"
        for filename,h in m["output_hashes"].items():
            path=directory/filename;assert sha(path)==h,filename
            hashes[str(path.relative_to(ROOT))]=h
        return m
    def check(verifier,*arguments):
        command=[sys.executable,str(Path(__file__).with_name(verifier)),*map(str,arguments)]
        if require_checkpoints:command.append("--require-checkpoints")
        subprocess.run(command,check=True,capture_output=True,text=True)
    for filename in ("analyze_budget.py","budget_outcomes.py","budget_statistics.py","budget_costs.py",
        "budget_test_outputs.py","verify_budget_test.py","verify_budget_calibration.py","verify_budget_behavior.py",
        "verify_budget_sft.py","verify_expanded_controls.py","verify_feasibility.py","verify_forward_ledger.py",
        "budget_instrument_verification.py","training_ledger.py","budget_protocol.py","budget_selection.py",
        "score_calibration.py","runtime.py"):
        path=Path(__file__).with_name(filename);hashes[str(path.relative_to(ROOT))]=sha(path)
    check("verify_expanded_controls.py",*[DIRECTORY/n for n in outcomes.POPULATION],"--require-eligible")
    calibration=DIRECTORY/"budget-calibration-v1";check("verify_budget_calibration.py",calibration)
    cm=run(calibration);source=read(calibration/"selection.json")
    source_ledger=read(calibration/"forward-ledger.json")
    constructions={};behavior_runs={};sft_runs={};test_runs={}
    for name in outcomes.POPULATION:
        constructions[name]=run(DIRECTORY/name)
        for stage,verifier,destination in (("behavior","verify_budget_behavior.py",behavior_runs),
                                           ("sft","verify_budget_sft.py",sft_runs)):
            directory=DIRECTORY/f"budget-{stage}-{name}-v1";check(verifier,directory)
            destination[name]=(directory,run(directory))
    for name in ["base",*outcomes.POPULATION]:
        directory=DIRECTORY/f"budget-test-{name}-v1"
        verify_test(directory,require_checkpoints);test_runs[name]=(directory,run(directory))
    holdout=track(DIRECTORY/"teacher-audit-holdout.json")["splits"]
    methods=(*outcomes.PRIMARY_METHODS,*(() if source["abstain"] else outcomes.DIAGNOSTICS),"own_code","sft")
    vectors={};base_vectors={}
    base_directory=test_runs["base"][0]
    for split in outcomes.SPLITS:
        rows=holdout[split]["rows"]
        base_vectors[split]=correctness(read(base_directory/f"{split}/method-base_reference.json"),rows)
        for name in outcomes.POPULATION:
            directory=test_runs[name][0]
            for method in methods:
                vectors[name,split,method]=correctness(read(directory/f"{split}/method-{method}.json"),rows)
    result=outcomes.summarize(vectors,base_vectors,abstained=source["abstain"])

    def total(ledger):return costs.phase_cost(ledger,[p["name"] for p in ledger["phases"]])
    def timing(m,phase_seconds):
        timed=sum(phase_seconds.values())
        assert 0<=timed<=m["elapsed_seconds"]+1e-6
        return dict(run_wall_seconds=m["elapsed_seconds"],timed_phase_seconds=phase_seconds,
                    remaining_run_wall_seconds=max(0,m["elapsed_seconds"]-timed),
                    prerequisite_verification_seconds=m.get("prerequisite_verification_seconds",0.))
    capture=read(calibration/"capture-costs.json")
    source_seconds={"reference-ordinary":capture["ordinary"]["seconds"],"reference-honest":capture["honest"]["seconds"]}
    for p in source_ledger["phases"][2:]:source_seconds[p["name"]]=read(calibration/f'{p["name"]}.json')["seconds"]
    source_cost=dict(actual=total(source_ledger),**timing(cm,source_seconds))
    behavioral_cost={};sft_cost={};test_cost={};independent={};inference=[source_cost["actual"]];training=[]
    for name in outcomes.POPULATION:
        directory,m=behavior_runs[name];ledger=read(directory/"forward-ledger.json")
        seconds={p["name"]:read(directory/f'policy-{p["name"]}.json')["seconds"] for p in ledger["phases"]}
        behavioral_cost[name]=dict(actual=total(ledger),cpu_selection_seconds=m["cpu_selection_seconds"],**timing(m,seconds))
        inference.append(behavioral_cost[name]["actual"])
        winners=read(directory/"selection.json")
        test_ledger=read(test_runs[name][0]/"forward-ledger.json")
        independent[name]=costs.method_costs(source_ledger,ledger,test_ledger,winners,source["abstain"])
        directory,m=sft_runs[name];ledger=read(directory/"training-ledger.json")
        saved=read(directory/"sft-costs.json")
        sft_cost[name]=dict(actual_top_level_training_forwards=total(ledger),optimizer=saved,
                           **timing(m,{"sft_training_and_checkpoint_save":saved["seconds"]}))
        training.append(sft_cost[name]["actual_top_level_training_forwards"])
    for name,(directory,m) in test_runs.items():
        ledger=read(directory/"forward-ledger.json");categories={};seconds={}
        for p in ledger["phases"]:
            phase=p["name"]
            if phase.startswith("instrument-"):
                category="instruments";file=directory/f"{phase}.json"
            else:
                split,label=phase.split("/",1);file=directory/f"{split}/forward-{label}.json"
                if label.startswith("prompt-"):category="shared_policy_outputs"
                elif label.startswith("random_write_"):category="random_write_diagnostics"
                elif label in ("final_only","context_only"):category="position_ablations"
                else:
                    assert label in ("raw","own_code","sft","base_reference")
                    category=label
            categories.setdefault(category,[]).append(phase);seconds[phase]=read(file)["seconds"]
        test_cost[name]=dict(actual=total(ledger),categories={c:costs.phase_cost(ledger,names) for c,names in categories.items()},
                             **timing(m,seconds))
        inference.append(test_cost[name]["actual"])
    summed=lambda rows:{field:sum(r[field] for r in rows) for field in costs.FIELDS}
    all_stage_times=[source_cost,*behavioral_cost.values(),*sft_cost.values(),*test_cost.values()]
    return dict(kind="complete_fixed_population_budget_audit",input_hashes=hashes,
        outcomes=result,source_fitting_forecasts=cm["forecasts"],
        base_reference_correct={s:int(v.sum()) for s,v in base_vectors.items()},
        costs=dict(source_fitting=source_cost,behavioral_fitting=behavioral_cost,sft_fitting=sft_cost,
            executed_test_runs=test_cost,independent_method_comparisons=independent,
            actual_inference_totals=summed(inference),actual_top_level_training_forward_totals=summed(training),
            model_run_wall_seconds=sum(r["run_wall_seconds"] for r in all_stage_times),
            prerequisite_verification_seconds=sum(r["prerequisite_verification_seconds"] for r in all_stage_times),
            construction_seconds={name:m["elapsed_seconds"] for name,m in constructions.items()},
            scope="Named audit model runs, with construction separate. Excludes prior experiments, engineering, downloads and unrecorded analysis overhead. Training totals exclude backward/recomputation FLOPs.",
            matching="Equal primary forward allowances; actual independent counts reported, including lower use from ordinary-policy reuse or source abstention. Fixed padding is not an optimized deployment comparison."),
        interpretation_limits=["Constructed models sharing one base, not independent frontier model samples.",
            "Behavioral comparison is limited to the fixed 22 policies, 28 decoders per policy and separate SFT recipe, not every possible elicitation strategy.",
            "Question-bootstrap intervals are descriptive and do not quantify model-family uncertainty.",
            "Four-choice score accuracy can differ from free-generation behavior.",
            "Control false positives and failed own-code recovery remain visible; no post-test model exclusion.",
            "Higher recovered accuracy alone is not evidence of an added auditing decision."])


def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--out",type=Path);group.add_argument("--verify",type=Path)
    parser.add_argument("--require-checkpoints",action="store_true");args=parser.parse_args()
    result=analysis(args.require_checkpoints)
    if args.verify:
        assert result==read(args.verify)
        print(json.dumps(dict(verified=True,primary_contrasts=len(result["outcomes"]["primary_contrasts"]))))
    else:
        assert not args.out.exists();args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        print(json.dumps(dict(output=str(args.out),failed_forecasts=result["outcomes"]["failed_forecasts"]),indent=2))


if __name__=="__main__":main()
