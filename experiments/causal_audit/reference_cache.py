"""Find pinned references in the verifying researcher's local Hub cache."""
from pathlib import Path
from reference_format import REFERENCES


def snapshot(reference,cache_root=None):
    spec=REFERENCES[reference]
    if cache_root is None:
        from huggingface_hub.constants import HF_HUB_CACHE
        cache_root=HF_HUB_CACHE
    path=Path(cache_root)/("models--"+spec["model"].replace("/","--"))/"snapshots"/spec["revision"]
    assert path.is_dir(),f"Pinned reference snapshot is not cached: {path}"
    return path
