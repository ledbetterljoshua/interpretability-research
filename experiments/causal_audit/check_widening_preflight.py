"""Exercise strict equivalence gates and independent tensor mutation detection."""
import json
import sys
import struct
import tempfile
from pathlib import Path
import numpy as np
import widening
from widening_preflight_validation import compare,check_tensor,check_weights


def main():
    config=json.loads((Path(__file__).resolve().parents[2]/'data/causal_audit/widening-config-inspection-v1/small_base-config.json').read_text())
    small=dict(config,hidden_size=8,intermediate_size=24,num_hidden_layers=3,head_dim=4,
               num_attention_heads=4,num_key_value_heads=2,vocab_size=23)
    large=dict(small,hidden_size=16,intermediate_size=48)
    rng=np.random.default_rng(1429);tested=0
    for name,shape in widening.shapes(small).items():
        a=rng.normal(size=shape).astype(np.float32);b=widening.widen_weight(name,a,small,large)
        check_tensor(name,a,b,small,large)
        b.flat[-1]+=1
        try:check_tensor(name,a,b,small,large)
        except AssertionError:tested+=1
        else:raise AssertionError('Missed transformed-tensor mutation: '+name)
    a=dict(logits=np.zeros((24,23),dtype=np.float32),residuals=np.zeros((3,24,8),dtype=np.float32))
    a['logits'][:,0]=1
    b=dict(logits=a['logits'].copy(),residuals=np.concatenate([a['residuals']]*2,axis=-1))
    assert compare(a,b,[0,1,2,3])['ready']
    b['logits']*=2
    bad=compare(a,b,[0,1,2,3]);assert bad['forecasts']['choice_predictions'] and not bad['ready']
    b['logits']=a['logits'].copy();b['residuals'][-1,-1,-1]=.0011
    bad=compare(a,b,[0,1,2,3]);assert not bad['forecasts']['all_block_residuals']
    b['residuals'][:]=0;b['logits'][0,-1]=.0011
    assert not compare(a,b,[0,1,2,3])['forecasts']['full_logits']
    b['logits'][0,-1]=np.nan
    try:compare(a,b,[0,1,2,3])
    except AssertionError:pass
    else:raise AssertionError('Missed nonfinite value')
    # Exercise actual safetensors serialization and raw BF16 decoding, including
    # a source mutation that leaves the native/expanded pair mutually consistent.
    from safetensors.numpy import save_file
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp);weights={};header={};payload=[];offset=0
        for name,shape in widening.shapes(small).items():
            original=rng.normal(size=shape).astype(np.float32)
            bits=(original.view(np.uint32)>>16).astype('<u2')
            weights[name]=(bits.astype(np.uint32)<<16).view(np.float32)
            data=bits.tobytes();payload.append(data)
            header[name]=dict(dtype='BF16',shape=list(shape),data_offsets=[offset,offset+len(data)])
            offset+=len(data)
        encoded=json.dumps(header).encode();encoded+=b' '*((-len(encoded))%8)
        raw_path=path/'raw.safetensors'
        raw_path.write_bytes(struct.pack('<Q',len(encoded))+encoded+b''.join(payload))
        save_file(weights,path/'native.safetensors')
        save_file(widening.widen_state(weights,small,large),path/'expanded.safetensors')
        receipt=check_weights(path/'native.safetensors',path/'expanded.safetensors',raw_path,small,large)
        assert receipt['native_matches_pinned_bf16'] and receipt['expanded_exactly_matches_transformation']
        data=bytearray(raw_path.read_bytes());data[-1]^=1;raw_path.write_bytes(data)
        try:check_weights(path/'native.safetensors',path/'expanded.safetensors',raw_path,small,large)
        except AssertionError:pass
        else:raise AssertionError('Missed raw BF16 checkpoint mutation')
    assert not any(n in sys.modules for n in ('torch','transformers','peft'))
    print(json.dumps(dict(verified=True,transformed_tensor_mutations_detected=tested,
        detects_argmax_preserving_logit_error=True,detects_last_block_second_half_error=True,
        detects_nonchoice_logit_error=True,detects_nonfinite=True,model_loaded=False),indent=2))


if __name__=='__main__':main()
