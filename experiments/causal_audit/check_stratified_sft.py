"""Test the all-nine fitting barrier and native diagnostic routing without an LM."""
import json
from pathlib import Path
import sys
import tempfile
from fit_stratified_sft import require_complete_behaviors,reference_instruments
from run_stratified_sft import existing_policy


def main():
    rejected=0
    with tempfile.TemporaryDirectory() as temp:
        paths=[Path(temp)/str(i) for i in range(9)]
        for path in paths:
            path.mkdir();(path/'run.json').write_text(json.dumps(dict(status='complete')))
        require_complete_behaviors(paths)
        for status in ('running','error','resource_stopped'):
            (paths[-1]/'run.json').write_text(json.dumps(dict(status=status)))
            try:require_complete_behaviors(paths)
            except AssertionError:rejected+=1
            else:raise AssertionError('Incomplete behavioral barrier accepted')
        (paths[-1]/'run.json').unlink()
        for candidate in (paths,paths[:-1],[*paths[:-1],paths[0]]):
            try:require_complete_behaviors(candidate)
            except AssertionError:rejected+=1
            else:raise AssertionError('Missing or duplicate behavior accepted')
        for status in ('running','error','resource_stopped'):
            (paths[-1]/'run.json').write_text(json.dumps(dict(status=status)))
            try:existing_policy(paths[-1],True)
            except AssertionError:rejected+=1
            else:raise AssertionError('Unsafe SFT restart accepted')
        (paths[-1]/'run.json').write_text(json.dumps(dict(status='complete')))
        assert existing_policy(paths[-1],True)=='verify-and-skip'
    assert reference_instruments('reference-widened-base').as_posix().endswith('widening-preflight-base-v1/expanded/instrument-logits.npz')
    for key in ('post','base'):
        assert reference_instruments('reference-'+key).as_posix().endswith(f'reference-preflight-{key}-v1/instrument-logits.npz')
    import verify_stratified_sft
    assert not any(n in sys.modules for n in ('torch','transformers','peft'))
    print(json.dumps(dict(verified=True,rejected_incomplete_or_unsafe_cases=rejected,reference_routes=3,model_loaded=False),indent=2))


if __name__=='__main__':main()
