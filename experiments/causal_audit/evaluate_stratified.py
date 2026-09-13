"""Frozen nine-model audit; no fitting or selection from reserved outcomes."""
from runtime import ROOT,Run,atomic_json,configure,sha
configure()
import argparse
import json
from pathlib import Path
import time
import numpy as np
import stratified_protocol as sp
import stratified_models as sm
from stratified_freeze import PLAN,CALIBRATION,DIRECTIONS,DATA,HOLDOUT,SPLITS,prerequisites
import budget_inference as bi
import budget_protocol as bp
import budget_interventions as edits
import budget_test_outputs as views
import interventions as it
from retarget_ledger import RetargetableForwardLedger


def read(path):return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('target',choices=sp.POPULATION)
    parser.add_argument('--check-prerequisites',action='store_true');args=parser.parse_args()
    started=time.monotonic();frozen=prerequisites();seconds=time.monotonic()-started
    if args.check_prerequisites:
        print(json.dumps(dict(verified=True,target=args.target,model_loaded=False,reserved_rows_read=False)));return
    # No reserved row is opened before all nineteen fits, full code and analysis
    # sources have passed the committed-evidence barrier above.
    holdout=read(HOLDOUT)['splits'];assert set(holdout)==set(SPLITS)
    assert all(len(holdout[s]['rows'])==256 for s in SPLITS)
    dev=read(DATA)['splits'];checks=dev['validation']['rows'][:4]
    canonical=[next(r for r in dev['train']['rows'] if r['answer']==i) for i in range(4)]
    selected=read(CALIBRATION/'selection.json');target=args.target
    behavior=read(ROOT/'data/causal_audit'/f'stratified-behavior-{target}-v1'/'selection.json')
    sft=ROOT/'data/causal_audit'/f'stratified-sft-{target}-v1';sft_manifest=read(sft/'run.json')
    local=sm.local_checkpoint_hashes(target)
    local.update({str((sft/n).relative_to(ROOT)):h for n,h in sft_manifest['sft_checkpoint_hashes'].items()})
    for name,digest in local.items():assert sha(ROOT/name)==digest,name
    sources=list(dict.fromkeys([*frozen,*[ROOT/n for n in local]]))
    native=target in sp.REFERENCES;widened=target=='reference-widened-base'
    own_code=None if native else read(ROOT/'data/causal_audit'/target/'run.json')['conditions']['unlock']
    projection=None if selected['abstain'] else read(DIRECTIONS/'directions.json')
    import torch
    from peft import PeftModel
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(1212)
    out=ROOT/'data/causal_audit'/f'stratified-test-{target}-v1'
    with Run(out,PLAN,sources,seconds=3600) as run:
        run.save(stage='loading',**sp.metadata(),**sm.descriptor(target),
            dtype='float32',device='mps',attention_implementation='eager',padding_length=512,batch_size=4,
            source_selection=selected,behavioral_winners={k:behavior[k] for k in ('prompt_only','decoded')},
            test_ids={s:[r['id'] for r in holdout[s]['rows']] for s in SPLITS},excluded={s:[] for s in SPLITS},
            instrument_ids=[r['id'] for r in checks],demonstration_ids=[r['id'] for r in canonical],
            prerequisite_verification_seconds=seconds,local_checkpoint_hashes=local,
            cached_file_hashes=sm.cached_file_hashes(target),own_code_prefix=own_code,
            random_direction_seeds=list(edits.RANDOM_SEEDS),graft_diagnostics_skipped=bool(selected['abstain']),
            widened_projection=projection['projection'] if widened and not selected['abstain'] else None,
            widened_projection_skipped=(not widened or selected['abstain'] or not projection['projection']['nondegenerate']))
        model,tokenizer=sm.load_original(target);choice_ids=tokenizer.choice_ids();table=[]
        with RetargetableForwardLedger(model) as ledger:
            def instrument(label):
                full={};run.save(stage='instrument',instrument=label)
                with ledger.phase('instrument-'+label,expected_examples=20):
                    result=bi.check_instruments(model,tokenizer,checks,choice_ids,full_logits_out=full)
                atomic_json(out/f'instrument-{label}.json',result)
                np.savez_compressed(out/f'instrument-{label}.npz',**{k:v.numpy() for k,v in full.items()})
                atomic_json(out/'forward-ledger.json',ledger.snapshot())
                assert result['passed'],f'Numerical instrument failed: {label}'

            def forward(split,label,prefix='',demonstrations=()):
                phase=f'{split}/{label}';run.save(stage='test',split=split,method=label)
                with ledger.phase(phase,expected_examples=256,sequence_length=512,batch_size=4):
                    result=bi.evaluate(model,tokenizer,holdout[split]['rows'],choice_ids,prefix,demonstrations,phase)
                filename=f'{split}/forward-{label}.json';atomic_json(out/filename,result)
                atomic_json(out/'forward-ledger.json',ledger.snapshot());return result,filename

            def view(split,label,pair,decoder=None,abstained=False):
                result=views.method_view(*pair,label,decoder,abstained)
                views.correctness(result,holdout[split]['rows']);atomic_json(out/f'{split}/method-{label}.json',result)
                table.append(dict(split=split,method=label,n=256,correct=result['correct'],abstained=abstained))
                atomic_json(out/'summary.json',table);print(json.dumps(dict(target=target,**table[-1])),flush=True)

            try:
                for split in SPLITS:(out/split).mkdir()
                instrument('original')
                directions=edits.random_writes();np.savez_compressed(out/'random-writes.npz',directions=directions)
                if not selected['abstain']:
                    layer=selected['layer']
                    with np.load(CALIBRATION/'vectors.npz',allow_pickle=False) as vectors:
                        read_vector=torch.tensor(vectors['unit'][layer],device='mps',dtype=torch.float32)
                        reference=float(vectors['reference'][layer])
                    assert abs(float(read_vector.norm())-1)<1e-6
                projected={}
                if widened and not selected['abstain']:
                    with np.load(DIRECTIONS/'directions.npz',allow_pickle=False) as arrays:
                        assert np.array_equal(arrays['read'],read_vector.cpu().numpy())
                        assert projection['source_layer']==layer and projection['source_reference']==reference
                        projected={k:torch.tensor(arrays[k],device='mps') for k in ('symmetric_write','symmetric_unit_write')}
                policies=list(dict.fromkeys(['ordinary',behavior['prompt_only']['policy'],behavior['decoded']['policy']]))
                for split in SPLITS:
                    cache={}
                    for name in policies:
                        policy=next(p for p in bp.BASELINE_POLICIES if p['name']==name)
                        cache[name]=forward(split,'prompt-'+name,policy['prefix'],bp.examples(policy,canonical))
                    view(split,'ordinary',cache['ordinary'])
                    for label in ('prompt_only','decoded'):
                        winner=behavior[label];view(split,label,cache[winner['policy']],winner['decoder'])
                    if selected['abstain']:view(split,'raw',cache['ordinary'],abstained=True)
                    else:
                        with it.graft(model,layer,read_vector,reference):view(split,'raw',forward(split,'raw'))
                        for seed,direction in zip(edits.RANDOM_SEEDS,directions):
                            label=f'random_write_{seed}'
                            with it.graft(model,layer,read_vector,reference,write=torch.tensor(direction,device='mps')):
                                view(split,label,forward(split,label))
                        for label in ('final_only','context_only'):
                            with edits.position_graft(model,layer,read_vector,reference,label):view(split,label,forward(split,label))
                        if widened:
                            for label,write in projected.items():
                                if projection['projection']['nondegenerate']:
                                    with it.graft(model,layer,read_vector,reference,write=write):view(split,label,forward(split,label))
                                else:view(split,label,cache['ordinary'],abstained=True)
                    if not native:view(split,'own_code',forward(split,'own_code',own_code))
                if native:
                    model=PeftModel.from_pretrained(model,sft/'checkpoints/sft',is_trainable=False).eval()
                    ledger.retarget(model)
                else:
                    model.load_adapter(sft/'checkpoints/sft',adapter_name='sft',is_trainable=False)
                    model.set_adapter('sft');model.eval()
                run.save(sft_state_event_index=len(ledger.events),outer_model_retargeted_for_sft=native)
                instrument('sft')
                for split in SPLITS:view(split,'sft',forward(split,'sft'))
            finally:atomic_json(out/'forward-ledger.json',ledger.snapshot())
        run.save(stage='finished',output_hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*')
            if p.is_file() and p.name!='run.json'})


if __name__=='__main__':main()
