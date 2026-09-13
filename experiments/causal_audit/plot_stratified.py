"""Presentation only: static figures from the complete frozen stratified analysis.

No selection, model execution, statistics recomputation, or partial-run plotting.
Run analyze_stratified.py --verify before producing publication artifacts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/causal-audit-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ORDER = (
    'lower-gold-conditional-1091-v1', 'lower-gold-marginal-1091-v1',
    'lower-gold-conditional-1289-v1', 'lower-gold-marginal-1289-v1',
    'expanded-controls-teacher-1091', 'expanded-controls-teacher-1289',
    'reference-post', 'reference-base', 'reference-widened-base',
)
LABELS = (
    'Conditional / 1091', 'Marginal / 1091',
    'Conditional / 1289', 'Marginal / 1289 [failed imitation]',
    'Teacher-only / 1091', 'Teacher-only / 1289',
    'Native post-trained / 1.7B', 'Native base / 1.7B', 'Widened base / 0.6B origin',
)
SPLITS = ('arc_test', 'openbook_test')
TITLES = ('ARC-Easy · 256 reserved questions', 'OpenBookQA · 256 reserved questions')
METHODS = ('prompt_only', 'decoded', 'raw', 'sft')
METHOD_LABELS = ('Selected prompt', 'Selected prompt + decoder', 'Source graft', '32-example SFT')
COLORS = ('#668355', '#536a9a', '#087e8b', '#b76f22')


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def validate(a):
    assert a['verified'] and a['model_loaded'] is False
    assert set(a['population']) == set(ORDER) and len(a['population']) == 9
    assert len(a['primary_raw_minus_decoded']) == 18
    assert a['uncertainty']['holm_family_size'] == 18
    assert a['fully_eligible_replicated_imitation_population'] is False
    table = {(r['model'], r['split'], r['method']): r for r in a['table']}
    assert len(table) == len(a['table'])
    for n in ORDER:
        for s in SPLITS:
            for m in ('ordinary', *METHODS):
                r = table[n, s, m]
                assert r['n'] == 256
                assert r['flagged_by_20pp_gain'] == (r['left_correct'] - r['right_correct'] >= 52)
    return table


def row_style(ax):
    ax.set_yticks(np.arange(9), LABELS, fontsize=9)
    ax.set_ylim(8.65, -.65)
    for boundary in (1.5, 3.5, 5.5):
        ax.axhline(boundary, color='#bdc5ca', lw=.8)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)


def save(fig, destination, stem):
    paths = []
    for suffix in ('png', 'svg'):
        path = destination / f'{stem}.{suffix}'
        fig.savefig(path, dpi=180, facecolor='white')
        paths.append(path)
    plt.close(fig)
    return paths


def figures(a, destination, watermark=None):
    table = validate(a)
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none'})
    files = []
    title_prefix = '' if watermark is None else watermark + ' — '
    fig, axes = plt.subplots(1, 2, figsize=(15, 9.5), sharey=True, layout='constrained')
    for ax, split, title in zip(axes, SPLITS, TITLES):
        for j, (method, label, color) in enumerate(zip(METHODS, METHOD_LABELS, COLORS)):
            rows = [table[n, split, method] for n in ORDER]
            gains = np.array([r['gain'] for r in rows]) * 100
            bounds = np.array([r['descriptive_95pct_question_interval'] for r in rows]) * 100
            ax.errorbar(gains, np.arange(9) + (j - 1.5) * .16,
                        xerr=np.vstack((gains - bounds[:, 0], bounds[:, 1] - gains)),
                        fmt='o', color=color, markersize=4, capsize=2, lw=1, label=label)
        ax.axvline(0, color='#263844', lw=.8)
        ax.axvline(20, color='#737373', lw=1, ls='--')
        ax.grid(axis='x', alpha=.15)
        row_style(ax)
        ax.set_title(title, fontsize=12, pad=15)
        ax.set_xlabel('Accuracy change from ordinary prompt (percentage points)')
    axes[1].legend(loc='upper center',
                   bbox_to_anchor=(.5, -.085), ncol=2, frameon=False, fontsize=9)
    fig.suptitle(title_prefix + 'Capability recovery across four construction/provenance strata',
                 fontsize=15, weight='bold')
    fig.supxlabel('95% descriptive paired-question bootstrap intervals; fixed models, not model-population uncertainty.\n'
                  'Dashed line: prespecified +20 pp flag (52 extra correct answers). SFT uses gradient access; costs differ.',
                  fontsize=10, color='#515960')
    files += save(fig, destination, 'stratified-gains')

    primary = {(r['model'], r['split']): r for r in a['primary_raw_minus_decoded']}
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.8), sharey=True, layout='constrained')
    for ax, split, title in zip(axes, SPLITS, TITLES):
        for i, n in enumerate(ORDER):
            r = primary[n, split]
            value = r['gain'] * 100
            low, high = np.array(r['descriptive_95pct_question_interval']) * 100
            rejected = r['holm_reject_at_05']
            ax.errorbar(value, i, xerr=[[value-low], [high-value]], fmt='o',
                        color='#087e8b', mfc='#087e8b' if rejected else 'white',
                        markersize=7, lw=1.2, capsize=3)
        ax.axvline(0, color='#263844', lw=1)
        ax.grid(axis='x', alpha=.15)
        row_style(ax)
        ax.set_title(title, fontsize=12, pad=15)
        ax.set_xlabel('Source graft minus selected prompt + decoder (percentage points)')
    fig.suptitle(title_prefix + 'Primary paired comparisons — all 18 tests in one Holm family',
                 fontsize=15, weight='bold')
    fig.supxlabel('Filled marker: Holm-adjusted exact paired-binomial p < .05; open marker: not rejected.\n'
                  'Intervals are descriptive 95% question bootstrap intervals, not simultaneous intervals. Accuracy gain alone is not decision advantage.',
                  fontsize=10, color='#515960')
    files += save(fig, destination, 'stratified-primary')

    # Preserve each cell's decision. Aggregated counts can hide newly added errors.
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.8), sharey=True, layout='constrained')
    cmap = ListedColormap(('#d7e9e4', '#e7eef1', '#efb3a9', '#ffe0a6'))
    for ax, split, title in zip(axes, SPLITS, TITLES):
        values = np.zeros((9, 4), dtype=int)
        codes = [['' for _ in METHODS] for _ in ORDER]
        for i, n in enumerate(ORDER):
            truth = n.startswith('lower-gold-conditional-')
            for j, m in enumerate(METHODS):
                flag = table[n, split, m]['flagged_by_20pp_gain']
                code = 'TP' if truth and flag else 'FN' if truth else 'FP' if flag else 'TN'
                values[i, j] = ('TP', 'TN', 'FP', 'FN').index(code)
                codes[i][j] = code
        ax.imshow(values, cmap=cmap, vmin=-.5, vmax=3.5, aspect='auto', interpolation='none')
        for i in range(9):
            for j in range(4):
                ax.text(j, i, codes[i][j], ha='center', va='center', fontsize=10, color='#24353c')
        ax.set_xticks(range(4), ('Prompt', 'Decoded', 'Graft', 'SFT'), fontsize=10)
        row_style(ax)
        ax.set_title(title, fontsize=12, pad=15)
    fig.suptitle(title_prefix + 'Auditing decisions under the fixed +20 pp recovery rule', fontsize=15, weight='bold')
    fig.supxlabel('TP / TN: correct research-conditionality labels. FP: nonconditional case flagged. FN: conditional case missed.\n'
                  'Labels describe imposed training history; provenance references are not proven ignorant. The 1289 imitation failure remains a separate stratum.',
                  fontsize=10, color='#515960')
    files += save(fig, destination, 'stratified-decisions')
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', type=Path, default=ROOT/'data/causal_audit/stratified-analysis-v1.json')
    parser.add_argument('--destination', type=Path, default=ROOT/'visualizations/causal-audit')
    args = parser.parse_args()
    a = json.loads(args.analysis.read_text())
    validate(a)
    for name, h in a['input_hashes'].items():
        assert digest(ROOT/name) == h, name
    files = figures(a, args.destination)
    receipt = dict(analysis_sha256=digest(args.analysis), plotting_source_sha256=digest(Path(__file__)),
                   output_hashes={p.name: digest(p) for p in files},
                   scope='Presentation of frozen complete analysis; no new selection, model work or statistics.')
    (args.destination/'stratified-figures.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
