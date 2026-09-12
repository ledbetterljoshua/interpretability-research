"""Known-vector and source-selection boundary checks, with no model weights."""
import json
import numpy as np
from budget_layers import LAYERS,select,vectors


def reject(call):
    try:call()
    except AssertionError:return
    raise AssertionError("Invalid selection input was accepted")


def main():
    a=np.zeros((28,2048),dtype=np.float32);b=a.copy()
    b[9,0]=3;b[9,1]=4
    unit,ref,metadata=vectors(a,b)
    assert np.allclose(unit[9,:2],[.6,.8]) and ref[9]==5
    assert metadata[0]==dict(layer=9,norm=5.,eligible=True)
    assert all(not r["eligible"] for r in metadata[1:])
    assert not unit[10:].any() and not ref[10:].any()
    table=[dict(layer=i,eligible=True,correct=4,gain=0) for i in LAYERS]
    table[1].update(correct=11,gain=7);table[2].update(correct=11,gain=7)
    table[3].update(eligible=False,correct=32,gain=28)
    answer=select(table,4);assert answer["layer"]==10 and not answer["abstain"]
    for r in table:
        if r["eligible"]:r.update(correct=10,gain=6)
    assert select(table,4)["abstain"]  # 6/32 is below the fixed 20 pp threshold.
    for r in table:r["eligible"]=False
    result=select(table,4);assert result["abstain"] and result["best_candidate"] is None
    reject(lambda:select(table[:-1],4))
    table[0]["gain"]=999;reject(lambda:select(table,4))
    print(json.dumps(dict(verified=True,model_weights_loaded=False,
        tied_winner=answer["layer"],six_of_32_abstains=True,degenerate_vectors_handled=True)))


if __name__=="__main__":main()
