"""Audit saved source provenance without importing experiment modules or an LM.

Checks static local Python imports, including imports inside functions. It does
not claim to discover dynamic imports, subprocess entry points, or dependencies
inside external libraries. The run's separate prerequisite verifiers cover
those experiment entry points and record external package versions.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / 'experiments/causal_audit'
ENTRY_POINTS = ('evaluate_stratified', 'verify_stratified_test', 'analyze_stratified')


def check(manifest):
    m = json.loads(manifest.read_text())
    seen = set()
    todo = list(ENTRY_POINTS)
    while todo:
        name = todo.pop()
        path = FOLDER / (name + '.py')
        if name in seen or not path.exists():
            continue
        seen.add(name)
        relative = str(path.relative_to(ROOT))
        assert relative in m['input_hashes'], f'Unrecorded local import: {relative}'
        source = path.read_bytes()
        assert hashlib.sha256(source).hexdigest() == m['input_hashes'][relative], relative
        for node in ast.walk(ast.parse(source)):
            candidates = ([node.module] if isinstance(node, ast.ImportFrom) else
                          [n.name for n in node.names] if isinstance(node, ast.Import) else [])
            todo.extend(n.split('.')[0] for n in candidates if n)
    return dict(verified=True, target=m['target'], observed_run_status=m['status'],
                entry_points=list(ENTRY_POINTS), static_local_import_count=len(seen),
                directly_hashed_modules=sorted(seen), model_loaded=False,
                scope='Source provenance only, not run completion or result verification.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.manifest), indent=2))
