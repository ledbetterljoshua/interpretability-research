"""Fixed 32-label LoRA SFT for unmodified references, with zero-edit validation."""
from runtime import ROOT,Run,atomic_json,configure,sha
configure()
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import reference_format as rf
import budget_inference as bi
import interventions as it
import elicitation
from forward_ledger import ForwardLedger
from training_ledger import TrainingLedger,verify_training_receipt

PLAN=ROOT/"notes/2026-09-12-causal-audit-budget-plan.md"
REFERENCE_PLAN=ROOT/"notes/2026-09-12-causal-audit-reference-budget-plan.md"
DATA=ROOT/"data/causal_audit/development.json"
POPULATION=[f"expanded-controls-{a}-{s}" for s in (1091,1289) for a in ("conditional","teacher","marginal")]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("reference",choices=list(rf.REFERENCES));args=parser.parse_args()
    assert PLAN.exists() and REFERENCE_PLAN.exists(),"Final main and reference budget plans are not committed yet"
    for plan in (PLAN,REFERENCE_PLAN):
        assert subprocess.check_output(["git","show",f"HEAD:{plan.relative_to(ROOT)}"],cwd=ROOT)==plan.read_bytes()
    started=time.monotonic()
    commands=[["verify_expanded_controls.py",*[str(ROOT/"data/causal_audit"/n) for n in POPULATION],
               "--require-eligible","--require-checkpoints"]]
    behaviors=[ROOT/"data/causal_audit"/f"budget-behavior-{n}-v1" for n in POPULATION]
    commands.extend(["verify_budget_behavior.py",str(p),"--require-checkpoints"] for p in behaviors)
    reference_behaviors=[ROOT/"data/causal_audit"/f"budget-behavior-reference-{key}-v1" for key in rf.REFERENCES]
    commands.extend(["verify_reference_behavior.py",str(p),"--require-weights"] for p in reference_behaviors)
    for command in commands:
        subprocess.run([sys.executable,str(Path(__file__).with_name(command[0])),*command[1:]],check=True,capture_output=True,text=True)
    prerequisite_seconds=time.monotonic()-started
    spec=rf.REFERENCES[args.reference];seed=1226+list(rf.REFERENCES).index(args.reference)
    preflight=ROOT/"data/causal_audit"/f"reference-preflight-{args.reference}-v1"
    original=json.loads((preflight/"run.json").read_text())
    validation=json.loads(DATA.read_text())["splits"]["validation"]["rows"];rows=validation[32:]
    sources=[Path(__file__),REFERENCE_PLAN,DATA,Path(rf.__file__),Path(bi.__file__),Path(it.__file__),Path(elicitation.__file__),
        preflight/"run.json",preflight/"instrument-logits.npz",
        *[p/"run.json" for p in behaviors+reference_behaviors],
        *[ROOT/"data/causal_audit"/n/"run.json" for n in POPULATION],
        *[Path(__file__).with_name(n) for n in ("runtime.py","precision.py","feasibility.py","training_ledger.py",
            "forward_ledger.py","verify_forward_ledger.py","verify_expanded_controls.py","verify_budget_behavior.py",
            "verify_budget_calibration.py","verify_feasibility.py","budget_protocol.py","verify_reference_sft.py",
            "verify_reference_behavior.py","verify_reference_preflight.py","budget_instrument_verification.py")]]
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import LoraConfig,get_peft_model
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(seed)
    out=ROOT/"data/causal_audit"/f"budget-sft-reference-{args.reference}-v1"
    with Run(out,PLAN,sources,seconds=1800) as run:
        tokenizer=rf.ReferenceTokenizer(AutoTokenizer.from_pretrained(spec["model"],revision=spec["revision"],local_files_only=True),args.reference)
        choice_ids=tokenizer.choice_ids();assert choice_ids==original["choice_ids"]
        encoded=[dict(id=r["id"],input_ids=tokenizer.encode(it.prompt(tokenizer,r)),target_token_id=choice_ids[r["answer"]]) for r in rows]
        atomic_json(out/"training-tokens.json",encoded)
        run.save(stage="loading",reference=args.reference,target=f"reference-{args.reference}",**spec,seed=seed,
            training_control_population=POPULATION,reference_population=list(rf.REFERENCES),
            cached_snapshot=original["cached_snapshot"],cached_file_hashes=original["cached_file_hashes"],
            dtype="float32",device="mps",attention_implementation="eager",training_padding="right_dynamic",
            choice_ids=choice_ids,adapter_key="default",adapter_initialization="new_zero_output_lora",
            training_ids=[r["id"] for r in rows],prerequisite_verification_seconds=prerequisite_seconds)
        base=AutoModelForCausalLM.from_pretrained(spec["model"],revision=spec["revision"],local_files_only=True,
            dtype=torch.float32,attn_implementation="eager").to("mps")
        model=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,
            target_modules=["q_proj","k_proj","v_proj","o_proj"],bias="none",task_type="CAUSAL_LM")).eval()
        trainable={name:p for name,p in model.named_parameters() if p.requires_grad}
        assert sum(p.numel() for p in trainable.values())==6422528
        assert all("lora_" in name and ".default." in name for name in trainable)
        b_weights={name:p for name,p in trainable.items() if ".lora_B." in name}
        assert len(b_weights)==112 and all(bool((p==0).all()) for p in b_weights.values())
        model.save_pretrained(out/"checkpoints/initial",selected_adapters=["default"])
        with ForwardLedger(model) as initialization_ledger:
            try:
                with initialization_ledger.phase("zero-adapter",expected_examples=4,sequence_length=512,batch_size=4):
                    with torch.no_grad():
                        logits=model(**bi.tokens_for(tokenizer,validation[:4]),use_cache=False,logits_to_keep=1).logits[:,-1].float().cpu().numpy()
                assert np.isfinite(logits).all()
                np.savez_compressed(out/"initialization-logits.npz",zero_adapter=logits)
                with np.load(preflight/"instrument-logits.npz",allow_pickle=False) as saved:reference=saved["padded"]
                error=float(np.abs(logits-reference).max())
                equal=bool(np.array_equal(logits[:,choice_ids].argmax(1),reference[:,choice_ids].argmax(1)))
                initial=dict(ids=[r["id"] for r in validation[:4]],b_matrices=112,b_matrices_all_zero=True,
                    trainable_parameters=6422528,max_full_logit_error=error,choice_predictions_equal=equal,
                    passed=error<.001 and equal,forward_examples=4,forward_calls=1,padded_token_positions=2048)
                atomic_json(out/"initialization.json",initial)
                assert initial["passed"],"Zero-output reference adapter changed numerical baseline"
            finally:atomic_json(out/"initialization-ledger.json",initialization_ledger.snapshot())
        with TrainingLedger(model) as ledger:
            try:
                with ledger.phase("sft-training",expected_examples=96,batch_size=4,require_inference=False):
                    costs=elicitation.train(model,tokenizer,rows,choice_ids,"default",out,run,seed)
                receipt=ledger.snapshot();verify_training_receipt(receipt)
                assert costs["training_input_tokens"]==sum(e["input_tokens"] for e in receipt["events"])
                atomic_json(out/"sft-costs.json",costs)
            finally:atomic_json(out/"training-ledger.json",ledger.snapshot())
        checkpoint_hashes=lambda label:{str(p.relative_to(out)):sha(p) for p in (out/"checkpoints"/label).rglob("*") if p.is_file()}
        initial_hashes=checkpoint_hashes("initial");final_hashes=checkpoint_hashes("sft")
        assert initial_hashes["checkpoints/initial/adapter_model.safetensors"]!=final_hashes["checkpoints/sft/adapter_model.safetensors"]
        run.save(stage="finished",costs=costs,initialization=initial,initial_checkpoint_hashes=initial_hashes,
            sft_checkpoint_hashes=final_hashes,
            output_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!="run.json"})
        print(json.dumps(dict(reference=args.reference,seed=seed,initialization=initial,costs=costs),indent=2),flush=True)


if __name__=="__main__":main()
