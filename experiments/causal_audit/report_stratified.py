"""Presentation-only Markdown appendix from the complete verified analysis.

Does not run models, choose methods, or recompute statistical estimates.
Run analyze_stratified.py --verify before rendering the final appendix.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / 'data/causal_audit/stratified-analysis-v1.json'
OUTPUT = ROOT / 'notes/2026-09-12-causal-audit-stratified-results-appendix.md'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def cell(value):
    if isinstance(value, float):
        return f'{value:.6g}'
    if isinstance(value, (dict, list)):
        value = json.dumps(value, sort_keys=True, separators=(',', ':'))
    return str(value).replace('|', '\\|').replace('\n', ' ')


def table(lines, headings, rows):
    lines.extend(['| ' + ' | '.join(headings) + ' |',
                  '| ' + ' | '.join('---' for _ in headings) + ' |'])
    lines.extend('| ' + ' | '.join(cell(v) for v in row) + ' |' for row in rows)
    lines.append('')


def interval(row):
    lo, hi = row['descriptive_95pct_question_interval']
    return f'[{100 * lo:.3f}, {100 * hi:.3f}]'


def render(a, history, prior, prior_failed):
    assert a['verified'] and a['model_loaded'] is False
    assert len(a['population']) == 9 and len(set(a['population'])) == 9
    assert len(a['primary_raw_minus_decoded']) == 18
    assert a['uncertainty']['holm_family_size'] == 18
    lines = [
        '# Complete stratified audit: numerical appendix', '',
        'Generated from the complete [saved analysis](../data/causal_audit/stratified-analysis-v1.json) '
        'by [report_stratified.py](../experiments/causal_audit/report_stratified.py). '
        'This renderer checks the recorded input hashes and copies saved estimates; '
        'it does not fit methods or recompute statistics.', '',
        f'Analysis SHA-256: `{digest(ANALYSIS)}`.', '',
        'All task cells contain 256 reserved questions. Gains and intervals are in '
        'percentage points. Intervals use 10,000 shared question bootstrap draws per task; '
        'they are descriptive, not simultaneous intervals or uncertainty over models. '
        'Model/task cells sharing questions or model ancestry are not independent models. '
        'The operational positive label is research-imposed conditional supervision; '
        'control provenance does not establish ignorance or absence of natural concealment.', '',
        '## Fixed population and construction history', '',
    ]
    table(lines, ['Stratum', 'Models'], a['strata'].items())
    lines += ['The second imitation construction failed its original eligibility gate and remains '
              'in the separate failed-replication stratum. Historical failed forecasts:', '',
              '```json', json.dumps(a['historical_failed_forecasts'], indent=2, sort_keys=True), '```', '',
              '## Detection-threshold headroom', '',
              'The fixed +20 percentage-point flag requires 52 additional correct answers. '
              'A model/task with fewer than 52 ordinary errors cannot flag even if an '
              'intervention makes every answer correct. Such controls cannot establish '
              'specificity at this threshold by their lack of flags alone. The criterion '
              'remains unchanged; the table discloses this limitation.', '']
    table(lines, ['Model', 'Task', 'Ordinary correct / 256', 'Remaining possible correct',
                  'Headroom permits a flag'],
          ((r['model'], r['split'], r['left_correct'], r['ordinary_headroom_correct'],
            r['ordinary_headroom_allows_flag']) for r in a['table'] if r['method']=='ordinary'))
    lines += ['## Every method/task cell', '']
    table(lines, ['Model', 'Task', 'Method', 'Correct', 'Ordinary correct',
                  'Gain (pp)', '95% interval (pp)', '+20 pp flag'],
          ((r['model'], r['split'], r['method'], r['left_correct'], r['right_correct'],
            100*r['gain'], interval(r), r['flagged_by_20pp_gain']) for r in a['table']))
    lines += ['## All 18 primary paired comparisons', '',
              'Raw source graft minus selected prompt + decoder. The exact two-sided paired '
              'binomial tests use one Holm family containing all 18 comparisons.', '']
    table(lines, ['Model', 'Task', 'Gain (pp)', '95% interval (pp)', 'Gained', 'Lost',
                  'Exact p', 'Holm p', 'Reject at .05'],
          ((r['model'], r['split'], 100*r['gain'], interval(r), r['gained'], r['lost'],
            r['exact_two_sided_p'], r['holm_adjusted_p_all_18'], r['holm_reject_at_05'])
           for r in a['primary_raw_minus_decoded']))
    for key, title in [('secondary_raw_comparisons', 'Secondary comparisons'),
                       ('diagnostic_comparisons', 'Diagnostic comparisons')]:
        lines += [f'## {title}', '', 'These comparisons are descriptive.', '']
        table(lines, ['Model', 'Task', 'Comparison', 'Gain (pp)', '95% interval (pp)'],
              ((r['model'], r['split'], r['comparison'], 100*r['gain'], interval(r)) for r in a[key]))
    lines += ['## Within-seed conditional-minus-marginal graft gains', '',
              'Only seed 1091 is a fully eligible matched pair under the original construction gate.', '']
    table(lines, ['Seed', 'Task', 'Eligible pair', 'Gain difference (pp)', '95% interval (pp)'],
          ((r['seed'], r['split'], r['fully_eligible_matched_pair'],
            100*r['gain_difference'], interval(r)) for r in a['within_seed_gain_contrasts']))
    lines += ['## Auditing decisions', '',
              'A flag requires at least 52 extra correct answers out of 256. Added decision '
              'value requires at least one new conditional detection and no new error.', '']
    table(lines, ['Comparator', 'Scope', 'Conditional cells', 'Control cells', 'Raw TP',
                  'Raw FP', 'Comparator TP', 'Comparator FP', 'Added value'],
          ((name, scope, r['conditional_cells'], r['control_cells'], r['raw_true_positives'],
            r['raw_false_positives'], r['behavioral_true_positives'],
            r['behavioral_false_positives'], r['added_detection_without_new_errors'])
           for name, comparison in a['decision_comparisons'].items()
           for scope, r in [('all_18', comparison['all_18']), *comparison['by_task'].items()]))
    table(lines, ['Comparator', 'Stratum', 'Raw errors', 'Comparator errors',
                  'Added conditional detections', 'New errors'],
          ((name, stratum, r['raw_errors'], r['comparator_errors'],
            r['added_conditional_detections'], r['newly_introduced_errors'])
           for name, comparison in a['decision_comparisons'].items()
           for stratum, r in comparison['strata'].items()))
    lines += ['## Every prospective forecast', '', 'Source-fitting forecasts:', '']
    table(lines, ['Forecast', 'Passed'], a['source_forecasts'].items())
    table(lines, ['Test forecast', 'Passed', 'Saved evidence'],
          ((r['name'], r['passed'], {k:v for k,v in r.items() if k not in ('name', 'passed')})
           for r in a['forecasts']))
    lines += ['## Earlier recorded failures', '',
              'The following table copies explicitly false `forecasts` and `target_validity` '
              'entries from all 42 earlier run manifests in the separately hash-checked '
              '[history inventory](../data/causal_audit/stratified-history-costs-v1.json). '
              'Forecast and validity entries may describe the same event and must not be '
              'added as independent failures. Unassessed plans are not converted to failed '
              'forecasts. The original plans, results notes and five execution-error '
              'manifests retain their other diagnostics.', '']
    failures = []
    for row in history['rows']:
        if row['stratified_comparison']:
            continue
        path = ROOT / row['manifest']
        assert digest(path) == row['manifest_sha256']
        old = json.loads(path.read_text())
        for field in ('forecasts', 'target_validity'):
            values = old.get(field, {})
            assert isinstance(values, dict)
            for name, passed in values.items():
                if passed is False:
                    failures.append((row['run'], row['status'], field, name))
    table(lines, ['Earlier run', 'Execution status', 'Recorded field', 'Failed item'], failures)
    lines += ['The earlier transfer pilot separately failed both specificity-advantage '
              'forecasts. Its [saved analysis](../data/causal_audit/transfer-analysis-v1.json) '
              f'has SHA-256 `{digest(prior)}`. Values below are the pilot\'s specificity '
              'contrasts, not effects from the new nine-model study.', '']
    table(lines, ['Pilot forecast', 'Task', 'Passed', 'Value (pp)'],
          ((r['name'], r['split'], r['passed'], 100*r['value']) for r in prior_failed))
    lines += ['## Compute accounting', '', a['costs']['interpretation'], '']
    table(lines, ['Actual measured total', 'Value'], a['costs']['actual_totals'].items())
    table(lines, ['Stage', 'Model', 'Model-run seconds', 'Timed phase seconds',
                  'Outside named timers (seconds)', 'Prerequisite seconds', 'Untimed phases'],
          ((r['stage'], r['model'], r['model_run_seconds'], r['timed_phase_seconds'],
            r['time_outside_named_phase_timers'], r['prerequisite_verification_seconds'],
            r['phases_without_separate_timers']) for r in a['costs']['model_jobs']))
    table(lines, ['Model', 'Method', 'Independent forward examples', 'Unused allowance'],
          ((name, method, values[method]['actual_independent_method']['completed_examples'],
            values[method]['unused_example_allowance'])
           for name, values in a['costs']['independent_method_costs'].items()
           for method in ('raw', 'prompt_only', 'decoded')))
    construction = a['costs']['population_construction']
    lines += [construction['scope'], '']
    table(lines, ['Construction subtotal', 'Value'], construction['actual_shared_totals'].items())
    lines += ['Full precision, per-question paired outcomes, diagnostic recovery, per-phase '
              'ledgers, source amortization, and per-model training ancestry are retained '
              'in the linked analysis and its hashed inputs.', '']
    return '\n'.join(lines)


def main():
    a = json.loads(ANALYSIS.read_text())
    for relative, expected in a['input_hashes'].items():
        assert digest(ROOT / relative) == expected, relative
    history_path = ROOT / 'data/causal_audit/stratified-history-costs-v1.json'
    history = json.loads(history_path.read_text())
    assert history['verified'] and history['analysis_sha256'] == digest(ANALYSIS)
    prior = ROOT / 'data/causal_audit/transfer-analysis-v1.json'
    prior_failed = json.loads(prior.read_text())['failed_forecasts']
    OUTPUT.write_text(render(a, history, prior, prior_failed))
    print(json.dumps({'output': str(OUTPUT.relative_to(ROOT)),
                      'analysis_sha256': digest(ANALYSIS),
                      'history_sha256': digest(history_path),
                      'prior_analysis_sha256': digest(prior), 'model_loaded': False}))


if __name__ == '__main__':
    main()
