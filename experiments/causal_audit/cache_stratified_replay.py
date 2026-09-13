"""Verify or download the three pinned public snapshots for experiment replay.

Default is local-only verification. --download permits missing public files to
be fetched. Neither path instantiates a model or edits experiment results.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = (
    'data/causal_audit/reference-preflight-post-v1/run.json',
    'data/causal_audit/reference-preflight-base-v1/run.json',
    'data/causal_audit/widening-reference-download-v1/download.json',
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '0' if args.download else '1'
    from huggingface_hub import snapshot_download
    receipts = []
    for relative in MANIFESTS:
        m = json.loads((ROOT / relative).read_text())
        expected = m['cached_file_hashes']
        snapshot = Path(snapshot_download(repo_id=m['model'], revision=m['revision'],
            allow_patterns=list(expected), local_files_only=not args.download,
            max_workers=1, token=False))
        assert snapshot.name == m['revision']
        for name, digest in expected.items():
            with (snapshot / name).open('rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest, name
        receipts.append(dict(model=m['model'], revision=m['revision'], files=len(expected)))
    assert not set(sys.modules).intersection({'torch', 'transformers', 'peft'})
    print(json.dumps(dict(verified=True, public_snapshots=receipts,
                         download_permitted=args.download, model_loaded=False), indent=2))


if __name__ == '__main__':
    main()
