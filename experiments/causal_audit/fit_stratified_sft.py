"""Fixed 24-update SFT comparators, after every behavioral fit is frozen."""
from runtime import ROOT,Run,atomic_json,configure,sha
configure()
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import stratified_protocol as sp
import stratified_models as sm
import budget_inference as bi
import interventions as it
import elicitation
from forward_ledger import ForwardLedger
from training_ledger import TrainingLedger,verify_training_receipt

PLAN=ROOT/sp.PLAN_NAME
DATA=ROOT/'data/causal_audit/development.json'
MODULES=('fit_stratified_sft.py','verify_stratified_sft.py','run_stratified_sft.py','check_stratified_sft.py',
    'stratified_models.py','verify_stratified_behavior.py','verify_budget_behavior.py','verify_budget_calibration.py',
    'verify_stratified_calibration.py','elicitation.py','interventions.py','budget_inference.py','budget_protocol.py',
    'runtime.py','precision.py','feasibility.py','training_ledger.py','forward_ledger.py','verify_forward_ledger.py',
    'verify_feasibility.py')


def reference_instruments(name):
    assert name in sp.REFERENCES
    path=ROOT/'data/causal_audit'/sp.REFERENCE_RUNS[name]
    return path/'expanded'/'instrument-logits.npz' if name=='reference-widened-base' else path/'instrument-logits.npz'


def require_complete_behaviors(behaviors):
    assert len(behaviors)==9 and len(set(behaviors))==9
    assert all((p/'run.json').exists() and json.loads((p/'run.json').read_text())['status']=='complete' for p in behaviors), \
        'All nine behavioral fits must be complete before SFT'


def prerequisites(name):
    assert PLAN.exists(),'Missing stratified plan'
    assert not any((ROOT/'data/causal_audit').glob('stratified-test-*-v1')),'SFT after test output is forbidden'
    behaviors=[ROOT/'data/causal_audit'/f'stratified-behavior-{n}-v1' for n in sp.POPULATION]
    require_complete_behaviors(behaviors)
    start=time.monotonic()
    from verify_stratified_population import verify as verify_population
    from verify_stratified_behavior import verify as verify_behavior
    assert verify_population(True)['verified']
    for path in behaviors:assert verify_behavior(path)['verified']
    local=sm.local_checkpoint_hashes(name)
    for path,digest in local.items():assert sha(ROOT/path)==digest,path
    sources=list(dict.fromkeys([*[Path(__file__).with_name(n) for n in MODULES],*sp.source_paths(ROOT),DATA,
        *[p/n for p in behaviors for n in ('run.json','selection.json')],*[ROOT/p for p in local],
        *([reference_instruments(name)] if name in sp.REFERENCES else [])]))
    for path in [PLAN,*sources]:
        if str(path.relative_to(ROOT)) not in local:
            assert subprocess.check_output(['git','show',f'HEAD:{path.relative_to(ROOT)}'],cwd=ROOT)==path.read_bytes(),path
    return sources,time.monotonic()-start


def main():
    parser=argparse.ArgumentParser();parser.add_argument('target',choices=sp.POPULATION)
    parser.add_argument('--check-prerequisites',action='store_true');args=parser.parse_args()
    sources,seconds=prerequisites(args.target)
    if args.check_prerequisites:
        assert not any(n in sys.modules for n in ('torch','transformers','peft'))
        print(json.dumps(dict(verified=True,target=args.target,model_loaded=False)));return
    validation=json.loads(DATA.read_text())['splits']['validation']['rows'];rows=validation[32:]
    seed=1220+sp.POPULATION.index(args.target);native=args.target in sp.REFERENCES
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(seed)
    out=ROOT/'data/causal_audit'/f'stratified-sft-{args.target}-v1'
    with Run(out,PLAN,sources,seconds=1800) as run:
        run.save(stage='loading',**sp.metadata(),**sm.descriptor(args.target),seed=seed,
            dtype='float32',device='mps',attention_implementation='eager',training_padding='right_dynamic',
            adapter_key='default',adapter_initialization='new_zero_output_lora' if native else 'continue_original_lora',
            training_ids=[r['id'] for r in rows],prerequisite_verification_seconds=seconds,
            local_checkpoint_hashes=sm.local_checkpoint_hashes(args.target),cached_file_hashes=sm.cached_file_hashes(args.target))
        model,tokenizer=sm.load_original(args.target,trainable=not native)
        choice_ids=tokenizer.choice_ids()
        encoded=[dict(id=r['id'],input_ids=tokenizer.encode(it.prompt(tokenizer,r)),target_token_id=choice_ids[r['answer']]) for r in rows]
        assert all(0<len(r['input_ids'])<=512 for r in encoded)
        atomic_json(out/'training-tokens.json',encoded)
        initial=None
        if native:
            from peft import LoraConfig,get_peft_model
            model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,
                target_modules=['q_proj','k_proj','v_proj','o_proj'],bias='none',task_type='CAUSAL_LM')).eval()
            trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
            assert sum(p.numel() for p in trainable.values())==6422528
            assert all('lora_' in n and '.default.' in n for n in trainable)
            b={n:p for n,p in trainable.items() if '.lora_B.' in n}
            assert len(b)==112 and all(bool((p==0).all()) for p in b.values())
            model.save_pretrained(out/'checkpoints/initial',selected_adapters=['default'])
            with ForwardLedger(model) as ledger:
                try:
                    with ledger.phase('zero-adapter',expected_examples=4,sequence_length=512,batch_size=4):
                        with torch.no_grad():
                            logits=model(**bi.tokens_for(tokenizer,validation[:4]),use_cache=False,logits_to_keep=1).logits[:,-1].float().cpu().numpy()
                    assert np.isfinite(logits).all()
                    np.savez_compressed(out/'initialization-logits.npz',zero_adapter=logits)
                    with np.load(reference_instruments(args.target),allow_pickle=False) as arrays:reference=arrays['padded']
                    error=float(np.abs(logits-reference).max())
                    equal=bool(np.array_equal(logits[:,choice_ids].argmax(1),reference[:,choice_ids].argmax(1)))
                    initial=dict(ids=[r['id'] for r in validation[:4]],b_matrices=112,b_matrices_all_zero=True,
                        trainable_parameters=6422528,max_full_logit_error=error,choice_predictions_equal=equal,
                        passed=error<.001 and equal,forward_examples=4,forward_calls=1,padded_token_positions=2048)
                    atomic_json(out/'initialization.json',initial)
                    assert initial['passed'],'Zero-output reference adapter changed the baseline'
                finally:atomic_json(out/'initialization-ledger.json',ledger.snapshot())
        with TrainingLedger(model) as ledger:
            try:
                with ledger.phase('sft-training',expected_examples=96,batch_size=4,require_inference=False):
                    costs=elicitation.train(model,tokenizer,rows,choice_ids,'default',out,run,seed)
                verify_training_receipt(ledger.snapshot())
                assert costs['training_input_tokens']==sum(e['input_tokens'] for e in ledger.snapshot()['events'])
                atomic_json(out/'sft-costs.json',costs)
            finally:atomic_json(out/'training-ledger.json',ledger.snapshot())
        checkpoint_hashes=lambda label:{str(p.relative_to(out)):sha(p) for p in (out/'checkpoints'/label).rglob('*') if p.is_file()}
        run.save(stage='finished',costs=costs,initialization=initial,
            initial_checkpoint_hashes=checkpoint_hashes('initial'),sft_checkpoint_hashes=checkpoint_hashes('sft'),
            output_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='run.json'})
        print(json.dumps(dict(target=args.target,seed=seed,costs=costs),indent=2),flush=True)


if __name__=='__main__':main()
