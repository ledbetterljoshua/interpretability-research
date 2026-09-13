"""Package exact local checkpoint dependencies; no model import or inference.

Archives are ignored by Git. The committed index records every member hash.
Public base checkpoints remain separate pinned Hugging Face downloads.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import sys
import stratified_protocol as sp

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'data/causal_audit/stratified-replay-v1'
INDEX = DEST / 'bundle.json'
WIDE = 'data/causal_audit/widening-preflight-base-v1/'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dependencies():
    files = {}
    sources = {}

    def read(path):
        sources[str(path.relative_to(ROOT))] = sha(path)
        value = json.loads(path.read_text())
        assert value['status'] == 'complete'
        return value

    def add(name, digest):
        path = PurePosixPath(name)
        assert not path.is_absolute() and '..' not in path.parts
        assert name.startswith('data/causal_audit/') and '/checkpoints/' in name
        assert files.setdefault(name, digest) == digest

    source = read(ROOT / 'data/causal_audit/stratified-calibration-v1/run.json')
    for name, digest in source['input_hashes'].items():
        if '/checkpoints/' in name:
            add(name, digest)
    for name in sp.POPULATION:
        for stage in ('behavior', 'sft'):
            directory = ROOT / 'data/causal_audit' / f'stratified-{stage}-{name}-v1'
            m = read(directory / 'run.json')
            for path, digest in m['input_hashes'].items():
                if '/checkpoints/' in path:
                    add(path, digest)
            if stage == 'sft':
                fields = ['sft_checkpoint_hashes']
                if name in sp.REFERENCES:
                    fields.append('initial_checkpoint_hashes')
                for field in fields:
                    for path, digest in m[field].items():
                        add(str((directory / path).relative_to(ROOT)), digest)
    widened = read(ROOT / WIDE / 'run.json')
    for name, digest in widened['checkpoint_hashes'].items():
        add(WIDE + name, digest)
    assert len(files) == 62, f'Unexpected checkpoint dependency count: {len(files)}'
    assert sum(name.endswith('.safetensors') for name in files) == 21
    sources[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    sources['experiments/causal_audit/stratified_protocol.py'] = sha(Path(__file__).with_name('stratified_protocol.py'))
    groups = {label: dict(sorted((n, h) for n, h in files.items() if n.startswith(WIDE) == wide))
              for label, wide in [('adapters', False), ('widening', True)]}
    assert len(groups['adapters']) == 57 and len(groups['widening']) == 5
    return sources, groups


def verify_archive(path, expected):
    seen = set()
    with tarfile.open(path, 'r:') as archive:
        for member in archive:
            assert member.isfile() and member.name in expected and member.name not in seen
            seen.add(member.name)
            with archive.extractfile(member) as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected[member.name]
    assert seen == set(expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    sources, groups = dependencies()
    if args.verify:
        index = json.loads(INDEX.read_text())
        assert index['source_hashes'] == sources
        assert set(index['archives']) == set(groups)
        for label, expected in groups.items():
            saved = index['archives'][label]
            assert saved['files'] == expected
            path = DEST / saved['path']
            assert path.stat().st_size == saved['bytes'] and sha(path) == saved['sha256']
            verify_archive(path, expected)
    else:
        assert not DEST.exists(), 'Replay output already exists; preserve it and use --verify'
        paths = list(sources)
        subprocess.run(['git', 'ls-files', '--error-unmatch', '--', *paths], cwd=ROOT,
                       check=True, capture_output=True)
        subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', *paths], cwd=ROOT,
                       check=True, capture_output=True)
        required = sum((ROOT / name).stat().st_size for group in groups.values() for name in group)
        assert shutil.disk_usage(ROOT).free > 2 * required + 10 * 2**30
        (DEST / 'checkpoints').mkdir(parents=True)
        index = dict(kind='exact_local_checkpoint_replay_bundle',
                     repository_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                     source_hashes=sources, archives={},
                     scope='Exact local checkpoint dependencies of source fitting, all 18 target fits, and widened preflight. Includes native SFT zero-adapter states for verification. The three public base-model snapshots and dataset caches are separate pinned downloads. This packages existing bytes; it neither repeats training nor proves inference reproducibility on another machine.')
        for label, expected in groups.items():
            relative = f'checkpoints/{label}.tar'
            path = DEST / relative
            with tarfile.open(path, 'w:') as archive:
                for name, digest in expected.items():
                    source = ROOT / name
                    assert sha(source) == digest, name
                    info = tarfile.TarInfo(name)
                    info.size = source.stat().st_size
                    info.mode = 0o644
                    info.mtime = 0
                    with source.open('rb') as stream:
                        archive.addfile(info, stream)
            verify_archive(path, expected)
            index['archives'][label] = dict(path=relative, bytes=path.stat().st_size,
                                            sha256=sha(path), files=expected)
            print(json.dumps(dict(archive=label, files=len(expected), bytes=path.stat().st_size,
                                  member_hashes_verified=True)), flush=True)
        INDEX.write_text(json.dumps(index, indent=2) + '\n')
    assert not set(sys.modules).intersection({'torch', 'transformers', 'peft'})
    print(json.dumps(dict(verified=True, files=62, archives=2, model_loaded=False)))


if __name__ == '__main__':
    main()
