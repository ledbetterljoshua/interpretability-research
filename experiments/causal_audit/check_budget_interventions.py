"""Known-state hook checks on a parameter-free CPU fixture; no model weights."""
from runtime import configure
configure()
import json
from types import SimpleNamespace
import numpy as np
import torch
from budget_interventions import random_writes,position_graft
from interventions import graft


def main():
    torch.set_num_threads(2)
    layer=torch.nn.Identity();model=SimpleNamespace(model=SimpleNamespace(layers=[layer]))
    states=torch.arange(40,dtype=torch.float32).reshape(2,5,4)/10
    before=states.clone();read=torch.tensor([.6,.8,0.,0.]);reference=1.3
    expected=states+(reference-states@read).unsqueeze(-1)*read
    with graft(model,0,read,reference):full=layer(states)
    assert torch.allclose(full,expected) and torch.equal(states,before)
    with position_graft(model,0,read,reference,"final_only"):final=layer(states)
    assert torch.equal(final[:,:-1],states[:,:-1]) and torch.allclose(final[:,-1],expected[:,-1])
    with position_graft(model,0,read,reference,"context_only"):context=layer(states)
    assert torch.equal(context[:,-1],states[:,-1]) and torch.allclose(context[:,:-1],expected[:,:-1])
    assert torch.allclose(final+context-states,full,atol=1e-6)
    with position_graft(model,0,read,reference,"context_only"):
        assert torch.equal(layer(states[:,:1]),states[:,:1])
    try:
        with position_graft(model,0,read,reference,"final_only"):
            raise RuntimeError("deliberate hook-scope failure")
    except RuntimeError:pass
    assert not layer._forward_hooks and torch.equal(layer(states),states)
    try:
        with position_graft(model,0,read,reference,"unknown"):pass
    except AssertionError:pass
    else:raise AssertionError("Unknown position policy accepted")
    directions=random_writes()
    assert directions.shape==(3,2048) and directions.dtype==np.float32
    assert np.array_equal(directions,random_writes())
    assert np.allclose(np.linalg.norm(directions.astype(np.float64),axis=1),1,rtol=0,atol=1e-7)
    assert len({v.tobytes() for v in directions})==3
    # At identical incoming states, all unit writes have the same displacement
    # norm within float32 tolerance, including a write distinct from the read.
    incoming=torch.linspace(-1,1,2048).reshape(1,1,2048)
    u=torch.tensor(directions[0]);norms=[]
    for direction in directions:
        with graft(model,0,u,1.3,write=torch.tensor(direction)):
            norms.append(float((layer(incoming)-incoming).norm()))
    assert max(norms)-min(norms)<1e-6
    assert not layer._forward_hooks
    print(json.dumps(dict(verified=True,model_weights_loaded=False,
        final_and_context_partition_verified=True,exception_cleanup_verified=True,
        random_direction_seeds=[1215,1216,1217],random_write_displacement_norms=norms)))


if __name__=="__main__":main()
