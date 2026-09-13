"""Predetermined projections of the frozen source direction; no target fitting."""
import numpy as np


def projected_writes(read):
    read=np.asarray(read)
    assert read.shape==(2048,) and read.dtype==np.float32 and np.isfinite(read).all()
    r=read.astype(np.float64);source_norm=float(np.linalg.norm(r))
    assert abs(source_norm-1)<1e-6,'Expected the frozen source unit direction'
    half=(r[:1024]+r[1024:])/2
    p=np.concatenate((half,half));anti=r-p
    norm=float(np.linalg.norm(p));eligible=norm>1e-8
    unit=p/norm if eligible else np.zeros_like(p)
    metadata=dict(source_norm=source_norm,symmetric_norm=norm,symmetric_energy=float(np.dot(p,p)),
        antisymmetric_energy=float(np.dot(anti,anti)),orthogonality_error=abs(float(np.dot(p,anti))),
        nondegenerate=eligible,diagnostics=('symmetric_write','symmetric_unit_write'),
        rule='Keep the source read/reference; project its write into [x,x], with and without restoring unit norm. No target selection.')
    assert metadata['orthogonality_error']<1e-12
    assert abs(metadata['symmetric_energy']+metadata['antisymmetric_energy']-source_norm**2)<1e-12
    return dict(read=read.copy(),symmetric_write=p.astype(np.float32),symmetric_unit_write=unit.astype(np.float32)),metadata
