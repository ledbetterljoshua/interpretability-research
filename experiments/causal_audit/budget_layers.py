"""Pure array construction and deterministic source-layer selection."""
import numpy as np

LAYERS=tuple(range(9,28))


def vectors(ordinary,honest):
    a=np.asarray(ordinary,dtype=np.float32);b=np.asarray(honest,dtype=np.float32)
    assert a.shape==b.shape==(28,2048) and np.isfinite(a).all() and np.isfinite(b).all()
    # Preserve the prior operator's float32 subtraction followed by float64
    # normalization. Store float32 vectors/references for the model edit.
    delta=(b-a).astype(np.float64)
    norms=np.linalg.norm(delta,axis=1)
    valid=norms>1e-8
    unit=np.zeros_like(delta);unit[valid]=delta[valid]/norms[valid,None]
    reference=np.einsum("ld,ld->l",unit,b.astype(np.float64))
    metadata=[dict(layer=i,norm=float(norms[i]),eligible=bool(valid[i])) for i in LAYERS]
    return unit.astype(np.float32),reference.astype(np.float32),metadata


def select(table,baseline_correct,n=32):
    assert n==32 and type(baseline_correct) is int and 0<=baseline_correct<=n
    assert [r["layer"] for r in table]==list(LAYERS),"Incomplete or reordered layer search"
    for r in table:
        assert type(r["eligible"]) is bool and type(r["correct"]) is int and 0<=r["correct"]<=n
        assert r["gain"]==r["correct"]-baseline_correct
    candidates=[r for r in table if r["eligible"]]
    best=max(candidates,key=lambda r:(r["gain"],-r["layer"])) if candidates else None
    abstain=best is None or best["gain"]/n<.2
    return dict(abstain=abstain,layer=None if abstain else best["layer"],
        best_candidate=best,n=n,baseline_correct=baseline_correct,
        rule="Maximum source accuracy gain; earlier layer on ties; abstain below 20 percentage points.")
