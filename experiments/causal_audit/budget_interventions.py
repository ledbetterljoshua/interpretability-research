"""Fixed-position ablations and random writes for the prospective audit.

No target fitting, model loading, or dataset access occurs in this module.
The all-position primary operator remains the original interventions.graft.
"""
from contextlib import contextmanager
import numpy as np
from interventions import decoder_layers

RANDOM_SEEDS=(1215,1216,1217)


def random_writes():
    result=[]
    for seed in RANDOM_SEEDS:
        rng=np.random.default_rng(seed)
        direction=rng.standard_normal(2048)
        direction/=np.linalg.norm(direction)
        result.append(direction.astype(np.float32))
    return np.stack(result)


@contextmanager
def position_graft(model,layer,read,reference,positions):
    assert positions in ("final_only","context_only")
    def hook(module,args,output):
        assert output.ndim==3 and output.shape[1]>0
        # Fixed left padding guarantees that the final position is real.
        selected=slice(-1,None) if positions=="final_only" else slice(None,-1)
        incoming=output[:,selected]
        result=output.clone()
        result[:,selected]=incoming+(reference-incoming@read).unsqueeze(-1)*read
        return result
    handle=decoder_layers(model)[layer].register_forward_hook(hook)
    try:yield
    finally:handle.remove()
