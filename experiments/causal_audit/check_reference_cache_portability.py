"""Verify a relocated tokenizer cache, then reject a corrupted relocated file."""
import json
from pathlib import Path
import tempfile
from runtime import ROOT
from reference_cache import snapshot
from reference_format import REFERENCES
from verify_reference_tokenizers import verify


def main():
    artifact=ROOT/"data/causal_audit/reference-tokenization-development-v1.json"
    original={key:snapshot(key) for key in REFERENCES}
    with tempfile.TemporaryDirectory(prefix="reference-cache-check-") as name:
        root=Path(name)
        for key,spec in REFERENCES.items():
            destination=root/("models--"+spec["model"].replace("/","--"))/"snapshots"/spec["revision"]
            destination.mkdir(parents=True)
            for p in original[key].iterdir():
                if p.is_file() and not p.name.endswith(".safetensors"):
                    (destination/p.name).symlink_to(p)
        result=verify(artifact,root,require_weights=False)
        assert result["verified"] and result["base_files_not_rehashed"]==["model.safetensors"]
        altered=snapshot("base",root)/"config.json"
        altered.unlink();altered.write_text("{}\n")
        try:verify(artifact,root,require_weights=False)
        except AssertionError:pass
        else:raise AssertionError("Corrupted relocated cache was accepted")
        # The original file is still intact: rejection depends on using the
        # supplied cache, not accidentally falling back to the recorded path.
        assert json.loads((original["base"]/"config.json").read_text())["hidden_size"]==2048
    print(json.dumps(dict(verified=True,model_weights_loaded=False,model_forwards=0,
        relocated_cache_without_weights_verified=True,corrupted_relocated_file_rejected=True,
        original_cache_unchanged=True)))


if __name__=="__main__":main()
