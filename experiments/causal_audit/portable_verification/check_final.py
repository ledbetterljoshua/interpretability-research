from pathlib import Path
import runpy
import sys
assert getattr(sys, '_causal_audit_portable_guard', False)
root = Path.cwd().resolve()
sys.path.insert(0, str(root/'experiments/causal_audit'))
for name in ('analyze_stratified.py', 'stratified_history_costs.py'):
    path = root/'experiments/causal_audit'/name
    sys.argv = [str(path), '--verify']
    runpy.run_path(str(path), run_name='__main__')
assert not set(sys.modules).intersection({'torch','transformers','peft','huggingface_hub','safetensors','tokenizers'})
print('FINAL_PORTABLE_VERIFICATION_PASSED')
