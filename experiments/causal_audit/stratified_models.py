"""Fixed original checkpoints and native formats; imports no LM until loading."""
import json
from pathlib import Path
import stratified_protocol as sp
from reference_format import REFERENCES,ReferenceTokenizer
from reference_cache import snapshot
from prepare_widening_reference import MODEL as SMALL_MODEL,REVISION as SMALL_REVISION

ROOT=Path(__file__).resolve().parents[2]


def read(path):return json.loads(path.read_text())


def descriptor(name):
    assert name in sp.POPULATION
    if name in sp.CONSTRUCTED:
        original=read(ROOT/'data/causal_audit'/name/'run.json')
        assert (original['model'],original['revision'])==(REFERENCES['post']['model'],REFERENCES['post']['revision'])
        return dict(target=name,model=original['model'],revision=original['revision'],format='chat',
            choice_ids=[32,33,34,35],adapter=f'data/causal_audit/{name}/checkpoints/final',width_expansion=1)
    if name=='reference-widened-base':
        return dict(target=name,model=SMALL_MODEL,revision=SMALL_REVISION,format='completion',
            choice_ids=[362,425,356,422],adapter=None,width_expansion=2)
    kind=name.removeprefix('reference-');spec=REFERENCES[kind]
    return dict(target=name,model=spec['model'],revision=spec['revision'],format=spec['format'],
        choice_ids=[32,33,34,35] if kind=='post' else [362,425,356,422],adapter=None,width_expansion=1)


def local_checkpoint_hashes(name):
    if name in sp.CONSTRUCTED:
        original=read(ROOT/'data/causal_audit'/name/'run.json')
        return {f'data/causal_audit/{name}/{p}':h for p,h in original['last_checkpoint_hashes'].items()}
    if name=='reference-widened-base':
        prefix='data/causal_audit/widening-preflight-base-v1/'
        m=read(ROOT/prefix/'run.json')
        return {prefix+p:h for p,h in m['checkpoint_hashes'].items() if p.startswith('checkpoints/expanded/')}
    return {}


def cached_file_hashes(name):
    if name in sp.CONSTRUCTED or name=='reference-post':
        original=read(ROOT/'data/causal_audit/reference-preflight-post-v1/run.json')
    elif name=='reference-base':
        original=read(ROOT/'data/causal_audit/reference-preflight-base-v1/run.json')
    else:original=read(ROOT/'data/causal_audit/widening-reference-download-v1/download.json')
    return original['cached_file_hashes']


def load_original(name,trainable=False):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    spec=descriptor(name)
    kind='post' if spec['format']=='chat' else 'base'
    if name=='reference-widened-base':
        from huggingface_hub.constants import HF_HUB_CACHE
        tokenizer_path=Path(HF_HUB_CACHE)/('models--'+SMALL_MODEL.replace('/','--'))/'snapshots'/SMALL_REVISION
        weights_path=ROOT/'data/causal_audit/widening-preflight-base-v1/checkpoints/expanded'
    else:tokenizer_path=weights_path=snapshot(kind)
    tokenizer=ReferenceTokenizer(AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True),kind)
    assert tokenizer.choice_ids()==spec['choice_ids'] and tokenizer.pad_token_id is not None
    base=AutoModelForCausalLM.from_pretrained(weights_path,local_files_only=True,dtype=torch.float32,
        attn_implementation='eager').to('mps')
    if spec['adapter'] is not None:
        from peft import PeftModel
        model=PeftModel.from_pretrained(base,ROOT/spec['adapter'],is_trainable=trainable)
    else:
        assert not trainable,'Native SFT must explicitly create and validate its zero-output LoRA'
        model=base
    return model.eval(),tokenizer
