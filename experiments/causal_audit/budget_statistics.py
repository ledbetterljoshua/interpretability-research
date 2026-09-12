"""Prospective paired-question summaries; no model or dataset loading.

Models are fixed constructed cases. Uncertainty here resamples questions only;
it cannot establish generalization to other models or natural concealment.
"""
import math
import numpy as np


def paired(left,right):
    a=np.asarray(left);b=np.asarray(right)
    assert a.ndim==b.ndim==1 and a.shape==b.shape and len(a)>0
    assert np.isin(a,[0,1]).all() and np.isin(b,[0,1]).all()
    delta=a.astype(np.int64)-b.astype(np.int64)
    wins=int((delta==1).sum());losses=int((delta==-1).sum());discordant=wins+losses
    # Conditional exact two-sided binomial test for paired binary outcomes.
    numerator=2*sum(math.comb(discordant,k) for k in range(min(wins,losses)+1))
    probability=min(1.,numerator/(1<<discordant))
    return dict(n=len(a),left_correct=int(a.sum()),right_correct=int(b.sum()),
        gain=float(delta.mean()),gained=wins,lost=losses,discordant=discordant,
        exact_two_sided_p=probability)


def holm(values):
    p=np.asarray(values,dtype=np.float64)
    assert p.ndim==1 and len(p)>0 and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    order=np.argsort(p,kind="stable");result=np.zeros_like(p);running=0.
    for rank,i in enumerate(order):
        running=max(running,min(1.,float(p[i])*(len(p)-rank)))
        result[i]=running
    return result.tolist()


def paired_intervals(values,seed=1213,draws=10000):
    """One shared question bootstrap across all columns, preserving dependence.

    Values are question-by-contrast, with model contrasts already formed. These
    are descriptive 95% percentile intervals, not simultaneous coverage claims.
    """
    x=np.asarray(values,dtype=np.float64)
    assert x.ndim==2 and len(x)>0 and x.shape[1]>0 and np.isfinite(x).all()
    assert type(draws) is int and draws>=100
    rng=np.random.default_rng(seed);samples=[]
    for start in range(0,draws,250):
        weights=rng.multinomial(len(x),np.full(len(x),1/len(x)),size=min(250,draws-start))
        samples.append(weights@x/len(x))
    estimates=np.concatenate(samples)
    return np.quantile(estimates,[.025,.975],axis=0).T.tolist()


def decision_comparison(raw,behavior,conditional):
    """Compare fixed flags in the same dataset/model order, without averaging away errors."""
    r=np.asarray(raw);b=np.asarray(behavior);y=np.asarray(conditional)
    assert r.ndim==b.ndim==y.ndim==1 and r.shape==b.shape==y.shape and len(y)>0
    assert all(np.isin(v,[False,True]).all() for v in (r,b,y))
    r=r.astype(bool);b=b.astype(bool);y=y.astype(bool)
    assert y.any() and (~y).any()
    added_true_positive=y&r&~b
    new_errors=(b==y)&(r!=y)
    return dict(cells=len(y),conditional_cells=int(y.sum()),control_cells=int((~y).sum()),
        raw_true_positives=int((y&r).sum()),raw_false_positives=int((~y&r).sum()),
        behavioral_true_positives=int((y&b).sum()),behavioral_false_positives=int((~y&b).sum()),
        added_true_positive_indices=np.flatnonzero(added_true_positive).tolist(),
        new_error_indices=np.flatnonzero(new_errors).tolist(),
        added_detection_without_new_errors=bool(added_true_positive.any() and not new_errors.any()))
