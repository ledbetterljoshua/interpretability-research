import os
import sys
BLOCKED = {'torch','transformers','peft','huggingface_hub','safetensors','tokenizers'}
class NoModelLibraries:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in BLOCKED:
            raise AssertionError('Model-library import attempted: '+fullname)
sys.meta_path.insert(0, NoModelLibraries())
def no_weight_reads(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(os.fspath(args[0]))
        if os.path.splitext(path)[1].lower() in {'.safetensors','.bin','.pt','.pth'}:
            raise AssertionError('Weight-file access attempted: '+path)
sys.addaudithook(no_weight_reads)
sys._causal_audit_portable_guard = True
