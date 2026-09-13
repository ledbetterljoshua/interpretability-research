"""Algebraic projection, pullback and norm checks, without any language model."""
import json
import sys
import numpy as np
from stratified_widened_diagnostics import projected_writes


def main():
    checked=0
    for seed in range(8):
        rng=np.random.default_rng(1501+seed)
        r=rng.normal(size=2048);r=(r/np.linalg.norm(r)).astype(np.float32)
        vectors,m=projected_writes(r);assert m['nondegenerate']
        rr=r.astype(np.float64);s=rr[:1024]+rr[1024:]
        p=np.concatenate((s/2,s/2));unit=p/np.linalg.norm(p)
        assert np.array_equal(vectors['symmetric_write'],p.astype(np.float32))
        assert np.array_equal(vectors['symmetric_unit_write'],unit.astype(np.float32))
        x=rng.normal(size=(7,1024));wide=np.concatenate((x,x),axis=-1);rho=.37
        delta=rho-wide@rr
        assert np.allclose(wide@rr,x@s,rtol=0,atol=1e-12)
        for write,scale in ((p,1.),(unit,np.linalg.norm(p))):
            edited=wide+delta[:,None]*write
            small=x+(rho-x@s)[:,None]*s/(2*scale)
            assert np.allclose(edited,np.concatenate((small,small),axis=-1),rtol=0,atol=1e-12)
        raw_delta=delta[:,None]*rr
        unit_delta=delta[:,None]*vectors['symmetric_unit_write'].astype(np.float64)
        assert np.max(np.abs(np.linalg.norm(raw_delta,axis=1)-np.linalg.norm(unit_delta,axis=1)))<1e-6
        checked+=1
    r=np.zeros(2048,dtype=np.float32);r[0]=r[1024]=np.float32(1/np.sqrt(2))
    vectors,m=projected_writes(r);assert m['nondegenerate'] and np.array_equal(vectors['symmetric_write'],r)
    r[1024]*=-1;vectors,m=projected_writes(r)
    assert not m['nondegenerate'] and not np.any(vectors['symmetric_write']) and not np.any(vectors['symmetric_unit_write'])
    for bad in (np.zeros(2048,dtype=np.float32),np.ones(1024,dtype=np.float32),np.full(2048,np.nan,dtype=np.float32)):
        try:projected_writes(bad)
        except AssertionError:pass
        else:raise AssertionError('Malformed direction accepted')
    assert not any(n in sys.modules for n in ('torch','transformers','peft'))
    print(json.dumps(dict(verified=True,random_cases=checked,native_pullbacks_verified=True,
        norm_matched_writes_verified=True,symmetric_and_antisymmetric_boundaries=True,model_loaded=False),indent=2))


if __name__=='__main__':main()
