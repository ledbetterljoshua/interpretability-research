"""Check all fixed model descriptors and reject destructive controller resumes."""
import json
from pathlib import Path
import sys
import tempfile
import stratified_protocol as sp
import stratified_models as sm
from run_stratified_behavior import existing_policy


def main():
    for name in sp.POPULATION:
        d=sm.descriptor(name);assert d['target']==name
        assert (d['format']=='chat')==(name in sp.CONSTRUCTED or name=='reference-post')
        assert d['choice_ids']==([32,33,34,35] if d['format']=='chat' else [362,425,356,422])
        assert (d['adapter'] is not None)==(name in sp.CONSTRUCTED)
        assert d['width_expansion']==(2 if name=='reference-widened-base' else 1)
        local=sm.local_checkpoint_hashes(name)
        expected=3 if name in sp.CONSTRUCTED else (2 if name=='reference-widened-base' else 0)
        assert len(local)==expected
        assert any(n.endswith('.safetensors') for n in sm.cached_file_hashes(name))
    rejected=0
    with tempfile.TemporaryDirectory() as temp:
        out=Path(temp)/'example';assert existing_policy(out,False)=='run'
        out.mkdir()
        for resume,status in ((False,None),(True,None),(True,'running'),(True,'error'),(True,'resource_stopped')):
            if status is not None:(out/'run.json').write_text(json.dumps(dict(status=status)))
            try:existing_policy(out,resume)
            except AssertionError:rejected+=1
            else:raise AssertionError('Unsafe existing output accepted')
        (out/'run.json').write_text(json.dumps(dict(status='complete')))
        assert existing_policy(out,True)=='verify-and-skip'
    assert not any(n in sys.modules for n in ('torch','transformers','peft'))
    print(json.dumps(dict(verified=True,model_descriptors=9,rejected_unsafe_resumes=rejected,model_loaded=False),indent=2))


if __name__=='__main__':main()
