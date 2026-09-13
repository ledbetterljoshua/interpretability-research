"""Retrospective cost disclosure for all recorded causal-audit run manifests.

Copies elapsed-time fields, including errors and rejected constructions. This
is not a re-verification of old model execution or a complete project bill.
Requires the finished nine-model analysis; never loads a language model.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / 'data/causal_audit/stratified-analysis-v1.json'
OUTPUT = ROOT / 'data/causal_audit/stratified-history-costs-v1.json'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build():
    analysis = json.loads(ANALYSIS.read_text())
    assert analysis['verified'] and len(analysis['population']) == 9
    assert len(analysis['costs']['model_jobs']) == 28
    for relative, expected in analysis['input_hashes'].items():
        assert sha(ROOT / relative) == expected, relative
    rows = []
    for path in sorted((ROOT / 'data/causal_audit').glob('*/run.json')):
        m = json.loads(path.read_text())
        assert m['status'] in ('complete', 'error'), f'Unfinished run: {path}'
        elapsed = m['elapsed_seconds']
        assert isinstance(elapsed, (int, float)) and math.isfinite(elapsed) and elapsed >= 0
        rows.append(dict(run=path.parent.name, manifest=str(path.relative_to(ROOT)),
                         manifest_sha256=sha(path), status=m['status'],
                         elapsed_seconds=elapsed,
                         stratified_comparison=path.parent.name.startswith('stratified-')))
    previous = [r for r in rows if not r['stratified_comparison']]
    current = [r for r in rows if r['stratified_comparison']]
    assert len(previous) == 42 and len(current) == 28, 'Unexpected historical or study inventory'
    assert sum(r['status'] == 'error' for r in previous) == 5
    assert all(r['status'] == 'complete' for r in current)
    expected = {'stratified-calibration-v1'} | {
        f'stratified-{stage}-{name}-v1'
        for stage in ('behavior', 'sft', 'test') for name in analysis['population']}
    assert {r['run'] for r in current} == expected
    current_sum = math.fsum(r['elapsed_seconds'] for r in current)
    assert math.isclose(current_sum, analysis['costs']['actual_totals']['model_run_seconds'],
                        rel_tol=0, abs_tol=1e-8)
    groups = {}
    for label, subset in [('earlier_runs', previous), ('stratified_comparison', current),
                          ('all_recorded_runs', rows)]:
        groups[label] = dict(runs=len(subset), complete=sum(r['status']=='complete' for r in subset),
                            error=sum(r['status']=='error' for r in subset),
                            recorded_elapsed_seconds=math.fsum(r['elapsed_seconds'] for r in subset))
    return dict(kind='retrospective_recorded_run_cost_inventory',
                source_sha256=sha(Path(__file__)), analysis_sha256=sha(ANALYSIS),
                rows=rows, totals=groups,
                scope='All 70 root-level data/causal_audit/*/run.json manifests present at the end of this study, each once. Includes rejected constructions and five execution errors. The 28 stratified-run sum agrees with the frozen analysis. This is a hash-checked transcription of recorded run elapsed fields, not independent timing evidence or verification of old model outputs.',
                exclusions='No claim to cover total project wall time, energy, FLOPs, token costs, all downloads, pretraining, engineering, waiting, or work that ended before creating a run manifest. Separately recorded prerequisite times are excluded. Do not add inherited training, phase timers, or the analysis construction subtotal again: those may already be represented by these run manifests.',
                verified=True, model_loaded=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    result = build()
    if args.verify:
        assert json.loads(OUTPUT.read_text()) == result, 'Saved cost inventory differs'
    else:
        assert not OUTPUT.exists(), 'Inventory exists; use --verify'
        OUTPUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(verified=True, totals=result['totals'], model_loaded=False), indent=2))


if __name__ == '__main__':
    main()
