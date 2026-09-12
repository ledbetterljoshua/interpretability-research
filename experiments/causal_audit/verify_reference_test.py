"""Reconstruct native reference audit outputs, frozen decisions and call costs."""
import audit_population as ap
import argparse
import json
from pathlib import Path
import numpy as np
from verify_feasibility import ROOT,sha
from verify_budget_calibration import read,evaluation
from verify_forward_ledger import verify as verify_ledger
from budget_instrument_verification import verify as verify_instrument
from verify_budget_test import predictions,IDENTITY,POPULATION,SPLITS
from reference_format import REFERENCES


def verify(out,require_checkpoints=False):
    m=read(out/"run.json");assert m["status"]=="complete"
    ap.require_provenance(m)
    key=m["reference"];assert key in REFERENCES and m["target"]==f"reference-{key}"
    assert all(m[k]==v for k,v in REFERENCES[key].items())
    assert m["training_control_population"]==POPULATION and m["reference_population"]==list(REFERENCES)
    assert m["original_adapter"] is None and m["ledger_hook_site"]=="outer_model_retargeted_for_sft"
    sft=ROOT/"data/causal_audit"/f"budget-sft-reference-{key}-v1";sft_manifest=read(sft/"run.json")
    checkpoint_paths={str((sft/n).relative_to(ROOT)) for n in sft_manifest["sft_checkpoint_hashes"]}
    missing=[]
    for name,h in m["input_hashes"].items():
        if not (ROOT/name).exists() and name in checkpoint_paths:missing.append(name)
        else:assert sha(ROOT/name)==h,name
    if require_checkpoints:assert not missing,missing
    for name,h in m["output_hashes"].items():assert sha(out/name)==h,name
    assert "notes/2026-09-12-causal-audit-reference-budget-plan.md" in m["input_hashes"]
    assert (m["dtype"],m["device"],m["attention_implementation"],m["padding_length"],m["batch_size"])==("float32","mps","eager",512,4)
    choice_ids=[32,33,34,35] if key=="post" else [362,425,356,422]
    assert m["choice_ids"]==choice_ids and m["prerequisite_verification_seconds"]>=0
    original=read(ROOT/"data/causal_audit"/f"reference-preflight-{key}-v1/run.json")
    assert m["cached_snapshot"]==original["cached_snapshot"] and m["cached_file_hashes"]==original["cached_file_hashes"]
    assert m["limits"]==dict(seconds=3600,rss_gib=32,mps_driver_gib=28,free_percent=15)
    assert m["elapsed_seconds"]<=3600 and m["peak_rss_gib"]<=32 and m["peak_mps_driver_gib"]<=28
    assert m["last_system_free_percent"]>=15
    holdout=ROOT/"data/causal_audit/teacher-audit-holdout.json"
    assert sha(holdout)=="8f0ca1b3bd765e53c2416140a0d65e32074a00408140ada50b194c37a723a98b"
    splits=read(holdout)["splits"];assert set(splits)==set(SPLITS)
    assert all(len(splits[s]["rows"])==256 for s in SPLITS)
    assert m["test_ids"]=={s:[r["id"] for r in splits[s]["rows"]] for s in SPLITS}
    assert m["excluded"]=={s:[] for s in SPLITS}
    dev=read(ROOT/"data/causal_audit/development.json")["splits"]
    ids=[r["id"] for r in dev["validation"]["rows"][:4]];assert m["instrument_ids"]==ids
    assert m["demonstration_ids"]==[next(r["id"] for r in dev["train"]["rows"] if r["answer"]==i) for i in range(4)]
    source=read(ROOT/"data/causal_audit/budget-calibration-v1/selection.json");assert m["source_selection"]==source
    for name in POPULATION:
        member=read(ROOT/"data/causal_audit"/name/"run.json")
        assert member["status"]=="complete" and member["eligible"] is True
    for name in POPULATION+[f"reference-{r}" for r in REFERENCES]:
        for stage in ("behavior","sft"):
            path=ROOT/"data/causal_audit"/f"budget-{stage}-{name}-v1/run.json"
            fitted=read(path);assert fitted["status"]=="complete" and fitted["target"]==name
            assert str(path.relative_to(ROOT)) in m["input_hashes"]
    behavior=read(ROOT/"data/causal_audit"/f"budget-behavior-reference-{key}-v1/selection.json")
    assert m["behavioral_winners"]=={k:behavior[k] for k in ("prompt_only","decoded")}
    assert m["graft_diagnostics_skipped"]==source["abstain"] and m["random_direction_seeds"]==[1215,1216,1217]
    ledger=read(out/"forward-ledger.json");receipt=verify_ledger(ledger)
    assert all(e["device"] in ("mps","mps:0") for e in ledger["events"])
    phases={p["name"]:p for p in ledger["phases"]};expected_phases=[]
    expected_outputs={"forward-ledger.json","summary.json","random-writes.npz"};table=[];instruments=[]

    def instrument(label):
        name=f"instrument-{label}";expected_phases.append(name);expected_outputs.update((name+".json",name+".npz"))
        with np.load(out/(name+".npz"),allow_pickle=False) as arrays:
            instruments.append(verify_instrument(read(out/(name+".json")),dict(arrays),phases[name],ledger["events"],ids,choice_ids))

    def forward(split,label):
        name=f"{split}/{label}";expected_phases.append(name)
        filename=f"{split}/forward-{label}.json";expected_outputs.add(filename)
        phase=phases[name]
        assert phase["completed_examples"]==phase["expected_examples"]==256 and phase["completed_calls"]==64
        assert phase["required_batch_size"]==4 and phase["required_sequence_length"]==512 and phase["require_inference"] is True
        return evaluation(out,name,splits[split]["rows"],choice_ids,phase,filename=filename),filename

    def view(split,label,pair,decoder=IDENTITY,abstained=False):
        raw,filename=pair;name=f"{split}/method-{label}.json";expected_outputs.add(name);result=read(out/name)
        assert (result["method"],result["n"],result["source_evaluation"],result["abstained"],result["additional_model_forwards"])==(label,256,filename,abstained,0)
        assert result["decoder"]==decoder
        expected=[dict(id=r["id"],answer=r["answer"],prediction=p) for r,p in zip(raw["records"],predictions(raw["records"],decoder))]
        assert result["records"]==expected
        correct=sum(r["prediction"]==r["answer"] for r in expected);assert result["correct"]==correct
        table.append(dict(split=split,method=label,n=256,correct=correct,abstained=abstained))

    with np.load(out/"random-writes.npz",allow_pickle=False) as directions:
        assert set(directions.files)=={"directions"};actual=directions["directions"]
    assert actual.shape==(3,2048) and actual.dtype==np.float32
    for i,seed in enumerate((1215,1216,1217)):
        v=np.random.default_rng(seed).standard_normal(2048);v/=np.linalg.norm(v)
        assert np.array_equal(actual[i],v.astype(np.float32))
    instrument("original")
    policies=list(dict.fromkeys(["ordinary",behavior["prompt_only"]["policy"],behavior["decoded"]["policy"]]))
    for split in SPLITS:
        cache={name:forward(split,f"prompt-{name}") for name in policies}
        view(split,"ordinary",cache["ordinary"])
        for label in ("prompt_only","decoded"):
            winner=behavior[label];view(split,label,cache[winner["policy"]],winner["decoder"])
        if source["abstain"]:view(split,"raw",cache["ordinary"],abstained=True)
        else:
            for label in ("raw","random_write_1215","random_write_1216","random_write_1217","final_only","context_only"):
                view(split,label,forward(split,label))
    assert m["sft_ledger_retarget_event_index"]==phases["instrument-sft"]["event_indices"][0]
    instrument("sft")
    for split in SPLITS:view(split,"sft",forward(split,"sft"))
    groups=len(policies)+(1 if source["abstain"] else 7)
    total_examples=groups*512+40;total_calls=groups*128+16
    assert list(phases)==expected_phases and set(m["output_hashes"])==expected_outputs
    assert table==read(out/"summary.json")
    assert receipt["completed_calls"]==total_calls and receipt["attempted_examples_known"]==total_examples
    assert receipt["attempted_padded_input_tokens_known"]==groups*512*512+sum(
        16*512+sum(i["ordinary_prompt_lengths"]) for i in instruments)
    return dict(verified=True,reference=key,method_dataset_cells=len(table),actual_forward_counts=receipt,
        instruments=instruments,checkpoint_files_unavailable=missing,
        scope="Saved native-format evidence, fixed decisions and actual-call receipts; no language-model rerun.")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("run",type=Path);parser.add_argument("--require-checkpoints",action="store_true")
    args=parser.parse_args();print(json.dumps(verify(args.run.resolve(),args.require_checkpoints),indent=2))


if __name__=="__main__":main()
