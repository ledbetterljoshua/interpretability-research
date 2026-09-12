"""Native-format instrument checks; no fresh test data or adapter loading."""
from runtime import ROOT, Run, atomic_json, configure, sha
configure()
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import reference_format as rf
import budget_inference as bi
import budget_protocol as bp
import interventions as it
from forward_ledger import ForwardLedger

PLAN=ROOT/"notes/2026-09-12-causal-audit-reference-preflight-plan.md"
DATA=ROOT/"data/causal_audit/development.json"
TOKENIZATION=ROOT/"data/causal_audit/reference-tokenization-development-v1.json"
DOWNLOAD=ROOT/"data/causal_audit/unmodified-reference-download-v1/download.json"


def main():
    parser=argparse.ArgumentParser();parser.add_argument("reference",choices=list(rf.REFERENCES));args=parser.parse_args()
    assert PLAN.exists(),"Reference preflight plan is missing"
    # Avoid taking the lock between jobs in the live six-model controller.
    construction_manifests=[ROOT/"data/causal_audit"/f"expanded-controls-{arm}-{seed}"/"run.json"
                            for seed in (1091,1289) for arm in ("conditional","teacher","marginal")]
    assert all(p.exists() and json.loads(p.read_text())["status"]=="complete" for p in construction_manifests), (
        "Wait for all six expanded construction jobs to finish before reference preflight")
    subprocess.run([sys.executable,str(Path(__file__).with_name("inspect_reference_tokenizers.py")),
                    "--verify",str(TOKENIZATION)],check=True,capture_output=True,text=True)
    spec=rf.REFERENCES[args.reference]
    cache=Path.home()/".cache/huggingface/hub"/("models--"+spec["model"].replace("/","--"))/"snapshots"/spec["revision"]
    files=[p for p in cache.iterdir() if p.is_file() and
           (p.suffix==".safetensors" or p.name in ("model.safetensors.index.json","config.json","generation_config.json",
                                                  "tokenizer.json","tokenizer_config.json","merges.txt","vocab.json"))]
    assert any(p.suffix==".safetensors" for p in files)
    cached_hashes={p.name:sha(p) for p in sorted(files)}
    d=json.loads(DATA.read_text())["splits"];rows=d["validation"]["rows"][:8]
    canonical=[next(r for r in d["train"]["rows"] if r["answer"]==i) for i in range(4)]
    sources=[Path(__file__),Path(rf.__file__),Path(bi.__file__),Path(bp.__file__),Path(it.__file__),
             DATA,TOKENIZATION,DOWNLOAD,*construction_manifests,
             *[Path(__file__).with_name(n) for n in ("runtime.py","forward_ledger.py","inspect_reference_tokenizers.py",
                 "verify_reference_preflight.py","verify_forward_ledger.py","budget_instrument_verification.py",
                 "verify_budget_calibration.py","verify_feasibility.py")]]
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(1230)
    out=ROOT/"data/causal_audit"/f"reference-preflight-{args.reference}-v1"
    with Run(out,PLAN,sources,seconds=600) as run:
        tokenizer=rf.ReferenceTokenizer(AutoTokenizer.from_pretrained(spec["model"],revision=spec["revision"],local_files_only=True),args.reference)
        choice_ids=tokenizer.choice_ids()
        expected=json.loads(TOKENIZATION.read_text())["references"][args.reference]
        assert choice_ids==expected["choice_ids"] and tokenizer.pad_token_id is not None
        policy=next(p for p in bp.BASELINE_POLICIES if p["name"]=="few_shot")
        lengths=[len(tokenizer.encode(it.prompt(tokenizer,r))) for r in rows]
        few_lengths=[len(tokenizer.encode(it.prompt(tokenizer,r,policy["prefix"],canonical))) for r in rows]
        assert max(lengths+few_lengths)<=512
        run.save(stage="loading",reference=args.reference,**spec,choice_ids=choice_ids,
            adapter=None,dtype="float32",device="mps",attention_implementation="eager",
            selected_ids=[r["id"] for r in rows],demonstration_ids=[r["id"] for r in canonical],
            ordinary_prompt_lengths=lengths,few_shot_prompt_lengths=few_lengths,
            cached_snapshot=str(cache),cached_file_hashes=cached_hashes)
        model=AutoModelForCausalLM.from_pretrained(spec["model"],revision=spec["revision"],local_files_only=True,
            dtype=torch.float32,attn_implementation="eager").to("mps").eval()
        with ForwardLedger(model) as ledger:
            try:
                raw={}
                with ledger.phase("instruments",expected_examples=20):
                    instruments=bi.check_instruments(model,tokenizer,rows[:4],choice_ids,full_logits_out=raw)
                np.savez_compressed(out/"instrument-logits.npz",**{k:v.numpy() for k,v in raw.items()})
                atomic_json(out/"instruments.json",instruments)
                assert instruments["passed"],"Native-reference numerical instrument forecast failed"
                results={}
                for label in ("ordinary-first","ordinary-repeat","few-shot"):
                    prefix=policy["prefix"] if label=="few-shot" else ""
                    examples=canonical if label=="few-shot" else ()
                    run.save(stage=label)
                    with ledger.phase(label,expected_examples=8,sequence_length=512,batch_size=4):
                        results[label]=bi.evaluate(model,tokenizer,rows,choice_ids,prefix,examples,label)
                    atomic_json(out/f"{label}.json",results[label])
                first,second=results["ordinary-first"],results["ordinary-repeat"]
                error=max(abs(a-b) for r,s in zip(first["records"],second["records"])
                          for a,b in zip(r["choice_logits"],s["choice_logits"]))
                predictions_equal=all(r["prediction"]==s["prediction"] for r,s in zip(first["records"],second["records"]))
                numerical=dict(instruments=instruments["passed"],repeated_logits=error<1e-5,
                               repeated_predictions=predictions_equal)
                valid=sum(r["top_is_choice"] for r in first["records"])
                performance=dict(ordinary_accuracy=first["correct"]>=(4 if args.reference=="post" else 2),
                                 ordinary_valid_top=valid>=4)
                summary=dict(repeated_choice_logit_max_error=error,numerical_forecasts=numerical,
                    performance_forecasts=performance,ready=all(numerical.values()),
                    ordinary_correct=first["correct"],ordinary_valid_top=valid,few_shot_correct=results["few-shot"]["correct"])
                atomic_json(out/"summary.json",summary)
                assert summary["ready"],"Native-reference repeatability forecast failed"
            finally:
                atomic_json(out/"forward-ledger.json",ledger.snapshot())
        run.save(stage="finished",summary=summary,
            output_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!="run.json"})
        print(json.dumps(dict(reference=args.reference,**summary),indent=2),flush=True)


if __name__=="__main__":main()
