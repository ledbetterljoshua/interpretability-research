"""Prospective held-out audit with frozen fits and explicit forward provenance.

Cannot execute without a final committed plan and all six eligible targets,
behavioral fits and SFT fits. One base/target job runs under the shared model lock.
"""
from runtime import ROOT,Run,atomic_json,configure,sha
configure()
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import budget_inference as bi
import budget_protocol as bp
import budget_selection as bs
import budget_interventions as edits
import budget_test_outputs as views
import interventions as it
from forward_ledger import ForwardLedger
from reference_freeze import require_reference_fits,REFERENCE_PLAN

PLAN=ROOT/"notes/2026-09-12-causal-audit-budget-plan.md"
DATA=ROOT/"data/causal_audit/development.json"
HOLDOUT=ROOT/"data/causal_audit/teacher-audit-holdout.json"
CALIBRATION=ROOT/"data/causal_audit/budget-calibration-v1"
POPULATION=[f"expanded-controls-{a}-{s}" for s in (1091,1289) for a in ("conditional","teacher","marginal")]
SPLITS=("arc_test","openbook_test")
MODEL="Qwen/Qwen3-1.7B"
REVISION="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def read(path):return json.loads(path.read_text())


def require_committed(paths):
    import hashlib
    for path in paths:
        name=str(path.relative_to(ROOT))
        committed=subprocess.check_output(["git","show",f"HEAD:{name}"],cwd=ROOT,stderr=subprocess.PIPE)
        assert hashlib.sha256(committed).hexdigest()==sha(path),f"Uncommitted fitting evidence: {name}"


def prerequisites():
    assert PLAN.exists(),"Final matched-forward audit plan is not committed yet"
    assert REFERENCE_PLAN.exists(),"Final reference budget plan is not committed yet"
    command=[sys.executable,str(Path(__file__).with_name("verify_expanded_controls.py")),
             *[str(ROOT/"data/causal_audit"/n) for n in POPULATION],"--require-eligible","--require-checkpoints"]
    subprocess.run(command,check=True,capture_output=True,text=True)
    stages=[("verify_budget_calibration.py",CALIBRATION)]
    stages.extend(("verify_budget_behavior.py",ROOT/"data/causal_audit"/f"budget-behavior-{n}-v1") for n in POPULATION)
    stages.extend(("verify_budget_sft.py",ROOT/"data/causal_audit"/f"budget-sft-{n}-v1") for n in POPULATION)
    frozen=[PLAN,Path(__file__),CALIBRATION/"selection.json",CALIBRATION/"vectors.npz"]
    for verifier,directory in stages:
        subprocess.run([sys.executable,str(Path(__file__).with_name(verifier)),str(directory),"--require-checkpoints"],
                       check=True,capture_output=True,text=True)
        frozen.append(directory/"run.json")
        if verifier=="verify_budget_behavior.py":frozen.append(directory/"selection.json")
    frozen.extend(ROOT/"data/causal_audit"/n/"run.json" for n in POPULATION)
    frozen.extend(require_reference_fits())
    require_committed(frozen)
    return frozen


def main():
    parser=argparse.ArgumentParser();parser.add_argument("target",choices=["base",*POPULATION]);args=parser.parse_args()
    started=time.monotonic();frozen=prerequisites();prerequisite_seconds=time.monotonic()-started
    assert sha(HOLDOUT)=="8f0ca1b3bd765e53c2416140a0d65e32074a00408140ada50b194c37a723a98b"
    require_committed([HOLDOUT])
    holdout=read(HOLDOUT)["splits"];assert set(holdout)==set(SPLITS)
    assert all(len(holdout[s]["rows"])==256 for s in SPLITS)
    dev=read(DATA)["splits"];checks=dev["validation"]["rows"][:4]
    canonical=[next(r for r in dev["train"]["rows"] if r["answer"]==i) for i in range(4)]
    selected=read(CALIBRATION/"selection.json")
    sources=[*frozen,DATA,HOLDOUT,Path(bi.__file__),Path(bp.__file__),Path(bs.__file__),Path(edits.__file__),
        Path(views.__file__),Path(it.__file__),
        *[Path(__file__).with_name(n) for n in ("runtime.py","forward_ledger.py","score_calibration.py",
            "verify_expanded_controls.py","verify_budget_calibration.py","verify_budget_behavior.py",
            "verify_budget_sft.py","verify_forward_ledger.py","verify_feasibility.py","training_ledger.py")]]
    if args.target!="base":
        target=ROOT/"data/causal_audit"/args.target;construction=read(target/"run.json")
        fit=ROOT/"data/causal_audit"/f"budget-behavior-{args.target}-v1"
        behavior=read(fit/"selection.json")
        sft=ROOT/"data/causal_audit"/f"budget-sft-{args.target}-v1";sft_manifest=read(sft/"run.json")
        sources.extend(target/n for n in construction["last_checkpoint_hashes"])
        sources.extend(sft/n for n in sft_manifest["sft_checkpoint_hashes"])
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(1212)
    out=ROOT/"data/causal_audit"/f"budget-test-{args.target}-v1"
    with Run(out,PLAN,sources,seconds=900 if args.target=="base" else 3600) as run:
        tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        encoded=[tokenizer.encode(c,add_special_tokens=False) for c in "ABCD"]
        assert all(len(x)==1 for x in encoded);choice_ids=[x[0] for x in encoded]
        run.save(stage="loading",model=MODEL,revision=REVISION,target=args.target,population=POPULATION,
            dtype="float32",device="mps",attention_implementation="eager",choice_ids=choice_ids,
            padding_length=512,batch_size=4,source_selection=selected,
            test_ids={s:[r["id"] for r in holdout[s]["rows"]] for s in SPLITS},
            excluded={s:[] for s in SPLITS},instrument_ids=[r["id"] for r in checks],
            demonstration_ids=[r["id"] for r in canonical],prerequisite_verification_seconds=prerequisite_seconds)
        base=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,
            dtype=torch.float32,attn_implementation="eager").to("mps").eval()
        model=base if args.target=="base" else PeftModel.from_pretrained(base,target/"checkpoints/final",is_trainable=False).eval()
        table=[]
        with ForwardLedger(model) as ledger:
            def instrument(label):
                full={};run.save(stage="instrument",instrument=label)
                with ledger.phase(f"instrument-{label}",expected_examples=20):
                    result=bi.check_instruments(model,tokenizer,checks,choice_ids,full_logits_out=full)
                atomic_json(out/f"instrument-{label}.json",result)
                np.savez_compressed(out/f"instrument-{label}.npz",**{k:v.numpy() for k,v in full.items()})
                atomic_json(out/"forward-ledger.json",ledger.snapshot())
                assert result["passed"],f"Instrument check failed: {label}"

            def forward(split,label,prefix="",demonstrations=()):
                phase=f"{split}/{label}";run.save(stage="test",split=split,method=label)
                with ledger.phase(phase,expected_examples=256,sequence_length=512,batch_size=4):
                    result=bi.evaluate(model,tokenizer,holdout[split]["rows"],choice_ids,
                        prefix=prefix,demonstrations=demonstrations,label=phase)
                filename=f"{split}/forward-{label}.json";atomic_json(out/filename,result)
                atomic_json(out/"forward-ledger.json",ledger.snapshot())
                return result,filename

            def view(split,label,pair,decoder=None,abstained=False):
                result=views.method_view(*pair,label,decoder,abstained)
                views.correctness(result,holdout[split]["rows"])
                atomic_json(out/f"{split}/method-{label}.json",result)
                table.append(dict(split=split,method=label,n=256,correct=result["correct"],abstained=abstained))
                atomic_json(out/"summary.json",table)
                print(json.dumps(dict(target=args.target,**table[-1])),flush=True)

            try:
                for split in SPLITS:(out/split).mkdir()
                instrument("base" if args.target=="base" else "original")
                if args.target=="base":
                    for split in SPLITS:view(split,"base_reference",forward(split,"base_reference"))
                else:
                    run.save(behavioral_winners={k:behavior[k] for k in ("prompt_only","decoded")},
                        own_code_prefix=construction["conditions"]["unlock"],random_direction_seeds=list(edits.RANDOM_SEEDS),
                        graft_diagnostics_skipped=bool(selected["abstain"]))
                    directions=edits.random_writes();np.savez_compressed(out/"random-writes.npz",directions=directions)
                    if not selected["abstain"]:
                        vectors=np.load(CALIBRATION/"vectors.npz",allow_pickle=False);layer=selected["layer"]
                        read_vector=torch.tensor(vectors["unit"][layer],dtype=torch.float32,device="mps")
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
                                with edits.position_graft(model,layer,read_vector,reference,label):
                                    view(split,label,forward(split,label))
                        view(split,"own_code",forward(split,"own_code",construction["conditions"]["unlock"]))
                    model.load_adapter(sft/"checkpoints/sft",adapter_name="sft",is_trainable=False)
                    model.set_adapter("sft");model.eval();instrument("sft")
                    for split in SPLITS:view(split,"sft",forward(split,"sft"))
            finally:
                atomic_json(out/"forward-ledger.json",ledger.snapshot())
        run.save(stage="finished",output_hashes={str(p.relative_to(out)):sha(p) for p in out.rglob("*")
                 if p.is_file() and p!=out/"run.json"})


if __name__=="__main__":main()
