"""Held-out native-reference audit; all eight models' fitting must be frozen."""
import audit_population as ap
from runtime import ROOT,Run,atomic_json,configure,sha
configure()
import argparse
import json
from pathlib import Path
import time
import numpy as np
import reference_format as rf
import budget_inference as bi
import budget_protocol as bp
import budget_selection as bs
import budget_interventions as edits
import budget_test_outputs as views
import interventions as it
from retarget_ledger import RetargetableForwardLedger
from evaluate_budget import prerequisites,require_committed,PLAN,DATA,HOLDOUT,CALIBRATION,SPLITS,POPULATION


def read(path):return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument("reference",choices=list(rf.REFERENCES));args=parser.parse_args()
    started=time.monotonic();frozen=prerequisites();prerequisite_seconds=time.monotonic()-started
    require_committed([Path(__file__),Path(__file__).with_name("verify_reference_test.py"),HOLDOUT])
    assert sha(HOLDOUT)=="8f0ca1b3bd765e53c2416140a0d65e32074a00408140ada50b194c37a723a98b"
    holdout=read(HOLDOUT)["splits"];assert set(holdout)==set(SPLITS)
    assert all(len(holdout[s]["rows"])==256 for s in SPLITS)
    dev=read(DATA)["splits"];checks=dev["validation"]["rows"][:4]
    canonical=[next(r for r in dev["train"]["rows"] if r["answer"]==i) for i in range(4)]
    selected=read(CALIBRATION/"selection.json");key=args.reference;spec=rf.REFERENCES[key]
    preflight=ROOT/"data/causal_audit"/f"reference-preflight-{key}-v1";original=read(preflight/"run.json")
    fit=ROOT/"data/causal_audit"/f"budget-behavior-reference-{key}-v1";behavior=read(fit/"selection.json")
    sft=ROOT/"data/causal_audit"/f"budget-sft-reference-{key}-v1";sft_manifest=read(sft/"run.json")
    sources=[*frozen,Path(__file__),DATA,HOLDOUT,Path(rf.__file__),Path(bi.__file__),Path(bp.__file__),Path(bs.__file__),
        Path(edits.__file__),Path(views.__file__),Path(it.__file__),
        *[sft/n for n in sft_manifest["sft_checkpoint_hashes"]],
        *[Path(__file__).with_name(n) for n in ("runtime.py","forward_ledger.py","score_calibration.py",
            "verify_audit_population.py","verify_budget_calibration.py","verify_budget_behavior.py","verify_budget_sft.py",
            "verify_forward_ledger.py","verify_feasibility.py","training_ledger.py","verify_reference_test.py",
            "verify_reference_preflight.py","budget_instrument_verification.py","retarget_ledger.py","verify_budget_test.py")]]
    sources.extend(ap.source_paths(ROOT))
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(1212)
    out=ROOT/"data/causal_audit"/f"budget-test-reference-{key}-v1"
    with Run(out,PLAN,sources,seconds=3600) as run:
        tokenizer=rf.ReferenceTokenizer(AutoTokenizer.from_pretrained(spec["model"],revision=spec["revision"],local_files_only=True),key)
        choice_ids=tokenizer.choice_ids();assert choice_ids==original["choice_ids"]
        run.save(stage="loading",reference=key,target=f"reference-{key}",**spec,
            training_control_population=POPULATION,reference_population=list(rf.REFERENCES),
            original_adapter=None,ledger_hook_site="outer_model_retargeted_for_sft",
            cached_snapshot=original["cached_snapshot"],cached_file_hashes=original["cached_file_hashes"],
            dtype="float32",device="mps",attention_implementation="eager",choice_ids=choice_ids,
            padding_length=512,batch_size=4,source_selection=selected,
            test_ids={s:[r["id"] for r in holdout[s]["rows"]] for s in SPLITS},excluded={s:[] for s in SPLITS},
            instrument_ids=[r["id"] for r in checks],demonstration_ids=[r["id"] for r in canonical],
            behavioral_winners={k:behavior[k] for k in ("prompt_only","decoded")},
            random_direction_seeds=list(edits.RANDOM_SEEDS),graft_diagnostics_skipped=bool(selected["abstain"]),
            prerequisite_verification_seconds=prerequisite_seconds)
        base=AutoModelForCausalLM.from_pretrained(spec["model"],revision=spec["revision"],local_files_only=True,
            dtype=torch.float32,attn_implementation="eager").to("mps").eval()
        model=base;table=[]
        with RetargetableForwardLedger(model) as ledger:
            def instrument(label):
                full={};run.save(stage="instrument",instrument=label)
                with ledger.phase(f"instrument-{label}",expected_examples=20):
                    result=bi.check_instruments(model,tokenizer,checks,choice_ids,full_logits_out=full)
                atomic_json(out/f"instrument-{label}.json",result)
                np.savez_compressed(out/f"instrument-{label}.npz",**{k:v.numpy() for k,v in full.items()})
                atomic_json(out/"forward-ledger.json",ledger.snapshot())
                assert result["passed"],f"Native reference instrument failed: {label}"

            def forward(split,label,prefix="",demonstrations=()):
                phase=f"{split}/{label}";run.save(stage="test",split=split,method=label)
                with ledger.phase(phase,expected_examples=256,sequence_length=512,batch_size=4):
                    result=bi.evaluate(model,tokenizer,holdout[split]["rows"],choice_ids,prefix,demonstrations,phase)
                filename=f"{split}/forward-{label}.json";atomic_json(out/filename,result)
                atomic_json(out/"forward-ledger.json",ledger.snapshot());return result,filename

            def view(split,label,pair,decoder=None,abstained=False):
                result=views.method_view(*pair,label,decoder,abstained);views.correctness(result,holdout[split]["rows"])
                atomic_json(out/f"{split}/method-{label}.json",result)
                table.append(dict(split=split,method=label,n=256,correct=result["correct"],abstained=abstained))
                atomic_json(out/"summary.json",table);print(json.dumps(dict(reference=key,**table[-1])),flush=True)

            try:
                for split in SPLITS:(out/split).mkdir()
                instrument("original")
                directions=edits.random_writes();np.savez_compressed(out/"random-writes.npz",directions=directions)
                if not selected["abstain"]:
                    with np.load(CALIBRATION/"vectors.npz",allow_pickle=False) as vectors:
                        layer=selected["layer"];read_vector=torch.tensor(vectors["unit"][layer],dtype=torch.float32,device="mps")
                        reference=float(vectors["reference"][layer])
                    assert abs(float(read_vector.norm())-1)<1e-6
                policy_names=list(dict.fromkeys(["ordinary",behavior["prompt_only"]["policy"],behavior["decoded"]["policy"]]))
                for split in SPLITS:
                    cache={}
                    for name in policy_names:
                        policy=next(p for p in bp.BASELINE_POLICIES if p["name"]==name)
                        cache[name]=forward(split,f"prompt-{name}",policy["prefix"],bp.examples(policy,canonical))
                    view(split,"ordinary",cache["ordinary"])
                    for label in ("prompt_only","decoded"):
                        winner=behavior[label];view(split,label,cache[winner["policy"]],winner["decoder"])
                    if selected["abstain"]:view(split,"raw",cache["ordinary"],abstained=True)
                    else:
                        with it.graft(model,layer,read_vector,reference):view(split,"raw",forward(split,"raw"))
                        for seed,direction in zip(edits.RANDOM_SEEDS,directions):
                            label=f"random_write_{seed}"
                            with it.graft(model,layer,read_vector,reference,write=torch.tensor(direction,device="mps")):
                                view(split,label,forward(split,label))
                        for label in ("final_only","context_only"):
                            with edits.position_graft(model,layer,read_vector,reference,label):view(split,label,forward(split,label))
                model=PeftModel.from_pretrained(base,sft/"checkpoints/sft",is_trainable=False).eval()
                ledger.retarget(model)
                run.save(sft_ledger_retarget_event_index=len(ledger.events))
                instrument("sft")
                for split in SPLITS:view(split,"sft",forward(split,"sft"))
            finally:atomic_json(out/"forward-ledger.json",ledger.snapshot())
        run.save(stage="finished",output_hashes={str(p.relative_to(out)):sha(p) for p in out.rglob("*")
                 if p.is_file() and p!=out/"run.json"})


if __name__=="__main__":main()
