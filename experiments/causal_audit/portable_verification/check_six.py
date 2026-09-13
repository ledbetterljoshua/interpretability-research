from pathlib import Path
import sys,json
assert getattr(sys,'_causal_audit_portable_guard',False)
root=Path.cwd().resolve();sys.path.insert(0,str(root/'experiments/causal_audit'))
import stratified_protocol as sp
from verify_stratified_test import verify
rows=[]
for name in sp.CONSTRUCTED:
    result=verify(root/'data/causal_audit'/f'stratified-test-{name}-v1')
    assert result['verified'] and len(result['checkpoint_files_unavailable'])==6
    rows.append(dict(model=name,forward_examples=result['actual_forward_counts']['attempted_examples_known'],missing_checkpoints=6))
assert len(rows)==6 and sum(r['forward_examples'] for r in rows)==32496
assert not set(sys.modules).intersection({'torch','transformers','peft','huggingface_hub','safetensors','tokenizers'})
print(json.dumps(dict(verified=True,tests=rows,model_imports_blocked=True,weight_file_access_blocked=True,model_loaded=False)))
