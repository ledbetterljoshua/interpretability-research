"""Save the two planned writes from the already frozen source; model-free."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from runtime import ROOT,sha,atomic_json
import stratified_protocol as sp
from stratified_widened_diagnostics import projected_writes

OUT=ROOT/'data/causal_audit/stratified-widened-directions-v1'
CALIBRATION=ROOT/'data/causal_audit/stratified-calibration-v1'
CAPTURES=ROOT/'data/causal_audit/widening-preflight-base-v1/expanded/captures.npz'
SOURCES=[ROOT/sp.PLAN_NAME,*[Path(__file__).with_name(n) for n in (
    'stratified_widened_diagnostics.py','check_stratified_widened_diagnostics.py',
    'prepare_stratified_widened_diagnostics.py','runtime.py','stratified_protocol.py')],
    *[CALIBRATION/n for n in ('run.json','selection.json','vectors.npz')],CAPTURES]


def reconstruct():
    source=json.loads((CALIBRATION/'run.json').read_text())
    assert source['status']=='complete'
    selection=json.loads((CALIBRATION/'selection.json').read_text());assert selection==source['selection']
    assert not selection['abstain'],'No projected diagnostic is defined when the primary method abstains'
    layer=selection['layer']
    with np.load(CALIBRATION/'vectors.npz',allow_pickle=False) as stored:
        vectors,metadata=projected_writes(stored['unit'][layer]);reference=float(stored['reference'][layer])
    with np.load(CAPTURES,allow_pickle=False) as stored:
        h=stored['residuals'];assert h.shape==(28,24,2048) and h.dtype==np.float32 and np.isfinite(h).all()
        half_error=float(np.max(np.abs(h[:,:,:1024]-h[:,:,1024:])))
    result=dict(verified=True,model_loaded=False,source_layer=layer,source_reference=reference,
        projection=metadata,observed_widened_preflight_half_max_error=half_error,
        forecasts=dict(projected_write_nondegenerate=metadata['nondegenerate']),
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in SOURCES})
    # Normalize tuples to the JSON representation used by the independent read.
    return vectors,json.loads(json.dumps(result))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    if not args.verify:
        for path in SOURCES:
            assert subprocess.check_output(['git','show',f'HEAD:{path.relative_to(ROOT)}'],cwd=ROOT)==path.read_bytes(),path
    vectors,result=reconstruct()
    if args.verify:
        stored=json.loads((OUT/'directions.json').read_text())
        assert stored['array_sha256']==sha(OUT/'directions.npz')
        assert {k:v for k,v in stored.items() if k!='array_sha256'}==result
        with np.load(OUT/'directions.npz',allow_pickle=False) as arrays:
            assert set(arrays.files)==set(vectors)
            for k in vectors:assert np.array_equal(arrays[k],vectors[k]),k
    else:
        OUT.mkdir(parents=True,exist_ok=False);np.savez_compressed(OUT/'directions.npz',**vectors)
        atomic_json(OUT/'directions.json',dict(**result,array_sha256=sha(OUT/'directions.npz')))
    assert not any(n in sys.modules for n in ('torch','transformers','peft'))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
