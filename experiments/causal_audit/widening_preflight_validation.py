"""Model-free numerical and independently sliced checkpoint checks."""
import json
import math
from pathlib import Path
import struct
import numpy as np
from widening import shapes, validate_configs


def compare(native, expanded, choice_ids):
    a, b = native['logits'], expanded['logits']
    x, y = native['residuals'], expanded['residuals']
    assert a.ndim == 2 and a.shape == b.shape
    assert x.ndim == y.ndim == 3 and x.shape[:2] == y.shape[:2]
    assert x.shape[1] == len(a) and y.shape[2] == 2*x.shape[2]
    assert all(v.dtype == np.float32 and np.isfinite(v).all() for v in (a,b,x,y))
    errors = np.max(np.abs(b-a), axis=1)
    d = x.shape[2]
    residual_errors = np.maximum(np.max(np.abs(y[:,:,:d]-x),axis=(1,2)),
                                 np.max(np.abs(y[:,:,d:]-x),axis=(1,2)))
    forecasts = dict(full_logits=float(errors.max()) < .001,
        choice_predictions=bool(np.array_equal(a[:,choice_ids].argmax(1),b[:,choice_ids].argmax(1))),
        all_block_residuals=float(residual_errors.max()) < .001)
    return dict(logit_max_absolute_error=float(errors.max()), case_logit_max_errors=errors.tolist(),
        residual_max_absolute_error=float(residual_errors.max()), block_residual_max_errors=residual_errors.tolist(),
        forecasts=forecasts, ready=all(forecasts.values()))


def check_tensor(name, original, expanded, small, large):
    """Check every output block without calling the construction function."""
    validate_configs(small,large)
    assert original.shape == shapes(small)[name] and expanded.shape == shapes(large)[name]
    assert original.dtype == expanded.dtype == np.float32
    assert np.isfinite(original).all() and np.isfinite(expanded).all()
    if name.endswith(('q_norm.weight','k_norm.weight')):
        assert np.array_equal(original,expanded),name
        return
    scale = .5 if name == 'model.norm.weight' or name.endswith(
        ('q_proj.weight','k_proj.weight','v_proj.weight','gate_proj.weight','up_proj.weight','down_proj.weight')) else 1.
    expected = original*scale
    if original.ndim == 1:
        n = len(original)
        assert np.array_equal(expanded[:n],expected) and np.array_equal(expanded[n:],expected),name
    else:
        nr,nc = original.shape
        assert expanded.shape[0]%nr == expanded.shape[1]%nc == 0
        for r in range(0,expanded.shape[0],nr):
            for c in range(0,expanded.shape[1],nc):
                assert np.array_equal(expanded[r:r+nr,c:c+nc],expected),name


def check_weights(native_path, expanded_path, raw_path, small, large):
    from safetensors import safe_open
    with raw_path.open('rb') as f:
        hlen = struct.unpack('<Q',f.read(8))[0]
        assert 0 < hlen < 4*1024*1024
        header = json.loads(f.read(hlen))
    assert set(header)-{'__metadata__'} == set(shapes(small))
    with safe_open(native_path,framework='np') as native, safe_open(expanded_path,framework='np') as expanded:
        assert set(native.keys()) == set(shapes(small))
        assert set(expanded.keys()) == set(shapes(large))
        for name, shape in shapes(small).items():
            a,b = native.get_tensor(name),expanded.get_tensor(name)
            check_tensor(name,a,b,small,large)
            h = header[name]; assert h['dtype'] == 'BF16' and tuple(h['shape']) == shape
            start,end = h['data_offsets']; assert end-start == 2*math.prod(shape)
            raw = np.memmap(raw_path,mode='r',dtype='<u2',offset=8+hlen+start,shape=shape)
            decoded = (np.asarray(raw).astype(np.uint32)<<16).view(np.float32)
            assert np.array_equal(a,decoded),name
            del a,b,raw,decoded
    return dict(unique_tensors=len(shapes(small)), native_matches_pinned_bf16=True,
        expanded_exactly_matches_transformation=True, dtype='float32', finite=True,
        native_parameters=sum(math.prod(s) for s in shapes(small).values()),
        expanded_parameters=sum(math.prod(s) for s in shapes(large).values()))
