"""Download and inspect the pinned small-base reference; never loads a model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import time
import widening

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/causal_audit/widening-reference-download-v1"
PLAN = ROOT / "notes/2026-09-12-causal-audit-widening-preflight-plan.md"
MODEL = "Qwen/Qwen3-0.6B-Base"
REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
FILES = ("config.json", "generation_config.json", "merges.txt", "model.safetensors",
         "tokenizer.json", "tokenizer_config.json", "vocab.json", "LICENSE", "README.md")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8*1024*1024), b""): digest.update(chunk)
    return digest.hexdigest()


def inspect(snapshot):
    config = json.loads((snapshot / "config.json").read_text())
    original = ROOT / "data/causal_audit/widening-config-inspection-v1/small_base-config.json"
    assert sha(snapshot / "config.json") == sha(original)
    weight_path = snapshot / "model.safetensors"
    with weight_path.open("rb") as file:
        length = struct.unpack("<Q", file.read(8))[0]
        assert 0 < length < 4*1024*1024
        header = json.loads(file.read(length))
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    expected = widening.shapes(config)
    assert set(tensors) == set(expected) and len(tensors) == 310
    intervals = []
    for name, shape in expected.items():
        row = tensors[name]
        assert row["dtype"] == "BF16" and tuple(row["shape"]) == shape, name
        start, end = row["data_offsets"]; assert end-start == 2*math.prod(shape), name
        intervals.append((start, end))
    intervals.sort(); assert intervals[0][0] == 0
    assert all(a[1] == b[0] for a,b in zip(intervals, intervals[1:]))
    assert weight_path.stat().st_size == 8 + length + intervals[-1][1]
    assert sum(math.prod(shape) for shape in expected.values()) == 596049920
    return dict(tensor_count=310, dtype="BF16", unique_parameters=596049920,
        stored_parameter_bytes=intervals[-1][1], header_bytes=length,
        file_bytes=weight_path.stat().st_size, duplicate_lm_head_absent=True,
        tensor_values_and_effective_model_tying_not_yet_checked=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--verify", action="store_true")
    parser.add_argument("--cache-root", type=Path); args = parser.parse_args()
    from huggingface_hub.constants import HF_HUB_CACHE
    cache_root = args.cache_root or Path(HF_HUB_CACHE)
    snapshot = cache_root / ("models--" + MODEL.replace("/", "--")) / "snapshots" / REVISION
    receipt = OUT / "download.json"
    if args.verify:
        m = json.loads(receipt.read_text()); assert m["status"] == "complete"
        assert (m["model"], m["revision"]) == (MODEL, REVISION)
        for name, digest in m["input_hashes"].items(): assert sha(ROOT / name) == digest, name
        for name, digest in m["cached_file_hashes"].items(): assert sha(snapshot / name) == digest, name
        assert m["header_inspection"] == inspect(snapshot)
        assert not any(n in sys.modules for n in ("torch", "transformers", "peft"))
        print(json.dumps(dict(verified=True, model_loaded=False, header=m["header_inspection"]), indent=2))
        return
    paths = [PLAN, Path(__file__), Path(widening.__file__),
        ROOT / "data/causal_audit/widening-config-inspection-v1/inspection.json",
        ROOT / "data/causal_audit/widening-config-inspection-v1/small_base-config.json",
        ROOT / "data/causal_audit/widening-algebra-check-v1.json"]
    for path in paths:
        assert subprocess.check_output(["git", "show", f"HEAD:{path.relative_to(ROOT)}"], cwd=ROOT) == path.read_bytes(), path
    OUT.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    m = dict(status="downloading", model=MODEL, revision=REVISION,
             input_hashes={str(p.relative_to(ROOT)): sha(p) for p in paths})
    def save():
        m["elapsed_seconds"] = time.monotonic()-started
        temp = receipt.with_suffix(".tmp"); temp.write_text(json.dumps(m, indent=2) + "\n"); temp.replace(receipt)
    save()
    try:
        from huggingface_hub import snapshot_download
        actual = Path(snapshot_download(MODEL, revision=REVISION, allow_patterns=list(FILES),
            cache_dir=cache_root, max_workers=2, token=False))
        assert actual.resolve() == snapshot.resolve()
        m["cached_file_hashes"] = {name: sha(snapshot / name) for name in FILES}
        m["cached_file_bytes"] = {name: (snapshot / name).stat().st_size for name in FILES}
        m["header_inspection"] = inspect(snapshot)
        assert not any(n in sys.modules for n in ("torch", "transformers", "peft"))
        m.update(status="complete", model_loaded=False, research_questions_read=False); save()
    except Exception as error:
        m.update(status="error", error=f"{type(error).__name__}: {error}"); save(); raise
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
