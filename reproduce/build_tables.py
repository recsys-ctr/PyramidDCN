"""Recompute Tables 1–4 and Welch/Holm statistics from saved scalar results."""
import argparse
import csv
import json
import math
import statistics as st
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from scipy.stats import ttest_ind
from .matrix_flops import matrix_flops
from .table3_statistics import TABLE3_SEEDS, paired_tests

ROOT = Path(__file__).resolve().parents[1]
DS = ['criteo', 'avazu', 'kkbox', 'kuairec']
SEEDS = [2024, 2025, 2026]
REF = 'pyramiddcn'
BASE = [('dcnv2', r'DCNv2~\cite{wang2021dcnv2}'),
        ('mldcn', r'ML-DCN~\cite{li2026mldcn}'), ('masknet', r'MaskNet~\cite{wang2021masknet}'),
        ('qnn', r'QNN-$\alpha$~\cite{li2025qnn}'), ('ecn', r'ECN~\cite{li2024fcn}'),
        ('fcn', r'FCN~\cite{li2024fcn}'), ('gdcn', r'GDCN~\cite{wang2023gdcn}'),
        ('finalmlp', r'FinalMLP~\cite{mao2023finalmlp}'), ('fibinetpp', r'FiBiNet++~\cite{zhang2023fibinetpp}')]


def fixed(value, places=5):
    return str(Decimal(str(value)).quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP))


def mean_text(values):
    exact = sum(map(lambda x: Decimal(str(x)), values), Decimal(0)) / Decimal(len(values))
    return str(exact.quantize(Decimal('.00001'), rounding=ROUND_HALF_UP))[1:]


def percent(value, places=2):
    q = Decimal(str(value)).quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP)
    return ('+' if q > 0 else '$-$' if q < 0 else '') + str(abs(q)) + r'\%'


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def holm(rows, field):
    previous = 0.
    for i, row in enumerate(sorted(rows, key=lambda x: x['p_raw'])):
        previous = max(previous, min(1., (len(rows) - i) * row['p_raw']))
        row[field] = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'outputs' / 'tables')
    args = parser.parse_args()
    out = args.out_dir.expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    records = {}
    for ds in DS:
        for line in (args.results_dir / f'{ds}.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row['name'] in records:
                raise ValueError(f'Duplicate run name: {row["name"]}')
            records[row['name']] = row
    def runs(ds, arm, seeds=SEEDS):
        prefix = f'{ds}_{arm}'
        cohort = [records[f'{prefix}_s{s}'] for s in seeds]
        if any(r['seed'] != seed or r['max_epochs'] != 1 for seed, r in zip(seeds, cohort)):
            raise ValueError(f'Invalid seed or epoch in cohort: {prefix}')
        return cohort
    def metric(rs, key='test_auc'):
        return [r[key] for r in rs]
    def mean(rs):
        return st.mean(metric(rs))
    comparisons = []
    for ds in DS:
        for key, _ in BASE:
            for m in ['test_auc', 'test_logloss']:
                x, y = metric(runs(ds, REF), m), metric(runs(ds, key), m)
                test = ttest_ind(x, y, equal_var=False)
                comparisons.append({'dataset': ds, 'baseline': key, 'metric': m, 'n_ours': len(x), 'n_baseline': len(y),
                                    'mean_ours': st.mean(x), 'mean_baseline': st.mean(y),
                                    'welch_t': float(test.statistic), 'welch_df': float(test.df), 'p_raw': float(test.pvalue)})
    stats = {}
    for m in ['test_auc', 'test_logloss']:
        family = [r for r in comparisons if r['metric'] == m]; holm(family, 'p_holm36')
        stats[m] = {'comparisons': len(family), 'max_p_holm36': max(r['p_holm36'] for r in family),
                    'all_significant': all(r['p_holm36'] < .05 for r in family)}
    write_csv(out / 'main_comparisons.csv', comparisons)
    (out / 'holm36_summary.json').write_text(json.dumps(stats, indent=2) + '\n')
    t1 = []
    for k, label in BASE + [('ours', r'\ours{}')]:
        cells = []
        for ds in DS:
            rs = runs(ds, REF) if k == 'ours' else runs(ds, k)
            for m, sign in [('test_auc', 1), ('test_logloss', -1)]:
                values = metric(rs, m)
                best = max([sign * st.mean(metric(runs(ds, b), m)) for b, _ in BASE] + [sign * st.mean(metric(runs(ds, REF), m))])
                number = mean_text(values)
                if sign * st.mean(values) >= best - 1e-12: number = r'\mathbf{' + number + '}'
                cells.append('$' + number + r'\,{\scriptstyle\pm' + f'{st.stdev(values):.5f}'[1:] + '}$')
            cells.append(f'{matrix_flops(rs[0])/1e6:.2f}M')
        if k == 'ours': t1.append(r'\midrule')
        t1.append(label + ' & ' + ' & '.join(cells) + r' \\')
    (out / 'table1_rows.tex').write_text('\n'.join(t1) + '\n')
    t2 = []; t2tests = []
    for k, label in BASE:
        # Table 2 starts from DCNv2 with rank W; Table 1 reports rank floor(W/4).
        original = runs('criteo', 'dcnv2_full_rank' if k == 'dcnv2' else k)
        variants = []
        for typ in ['pyramid', 'pyramid_matched_flops']:
            rs = runs('criteo', f'{k}_{typ}'); test = ttest_ind(metric(rs), metric(original), equal_var=False)
            t2tests.append({'model': k, 'configuration': typ, 'delta_auc': mean(rs)-mean(original),
                            't': float(test.statistic), 'df': float(test.df), 'p_raw': float(test.pvalue)})
            variants.append((rs, float(test.pvalue)))
        p, q = [r for r, _ in variants]
        star = lambda index: r'$^*$' if variants[index][1] < .05 else ''
        short = label.split(r'~\cite')[0]
        cells = [short, mean_text(metric(original)), f'{matrix_flops(original[0])/1e6:.2f}',
                 mean_text(metric(p)) + star(0), percent(100*(mean(p)/mean(original)-1)),
                 f'{matrix_flops(p[0])/1e6:.2f}', percent(100*(matrix_flops(p[0])/matrix_flops(original[0])-1), 0),
                 mean_text(metric(q)) + star(1), percent(100*(mean(q)/mean(original)-1))]
        t2.append(' & '.join(cells) + r' \\')
    ref = runs('criteo', REF)
    t2 += [r'\midrule', r'\ours{} & \textbf{' + mean_text(metric(ref)) + r'} & \textbf{' + f'{matrix_flops(ref[0])/1e6:.2f}' + r'} & & & & & & \\']
    (out / 'table2_rows.tex').write_text('\n'.join(t2) + '\n')
    write_csv(out / 'table2_welch.csv', t2tests)
    specs = json.loads((ROOT / 'configs' / 'table3_specification.json').read_text())
    t3 = []
    for spec in specs:
        arm = spec['arm']
        seeds = TABLE3_SEEDS
        def aggregate(a, shared):
            aa = {ds: runs(ds, a, shared) for ds in DS}; bb = {ds: runs(ds, REF, shared) for ds in DS}
            # Normalize by each dataset's mean reference, then average over the four fixed datasets.
            delta = 100 * st.mean(mean(aa[ds]) / mean(bb[ds]) - 1 for ds in DS)
            cost = st.mean(matrix_flops(aa[ds][0])/matrix_flops(bb[ds][0]) for ds in DS)
            return delta, cost
        delta, cost = aggregate(arm, seeds)
        matched = spec['matched_arm']; matched_delta = ''
        if matched:
            shared = TABLE3_SEEDS
            matched_delta = aggregate(matched, shared)[0]
        t3.append(dict(spec, seeds=seeds, flops_ratio=round(cost, 2), delta_auc_percent=delta, matched_delta_auc_percent=matched_delta))
    core_arms = json.loads((ROOT / 'configs' / 'experiment_index.json').read_text())['table3_arms']
    core = [next(r for r in t3 if r['arm'] == a) for a in core_arms]
    tests, seed_rows = paired_tests(runs, core, DS, REF)
    lookup = {(r['core_arm'], r['comparison']): r for r in tests}
    for row in core:
        row['p_holm13'] = lookup.get((row['arm'], 'same_widths'), {}).get('p_holm13', '')
        row['matched_p_holm13'] = lookup.get((row['arm'], 'matched_FLOPs'), {}).get('p_holm13', '')
    write_csv(out / 'table3.csv', core)
    write_csv(out / 'table3_significance.csv', tests)
    write_csv(out / 'table3_seed_averages.csv', seed_rows)
    table3_stats = {'seeds': TABLE3_SEEDS, 'datasets': DS, 'comparisons': len(tests),
                    'significant_holm05': sum(t['significant_holm05'] for t in tests),
                    'method': 'Two-sided paired t-tests on eight seed-wise normalized four-dataset averages; Holm across 13 comparisons.'}
    (out / 'table3_significance_summary.json').write_text(json.dumps(table3_stats, indent=2) + '\n')
    # Formula columns use the explicit published definitions; numeric cells are all recomputed above.
    formats = [(r'\textbf{\ours{}}', r's_i\had c_i+s_i'),
               ('Input pooling', r'u_i\had c_i+u_i'),
               ('Frozen projection', r'f_i\had c_i+f_i'),
               ('Learned projection', r'\ell_i\had c_i+\ell_i'),
               (r'Previous-layer channels', r't_i\had c_i+t_i'),
               ('Addition only', r's_i+c_i'),
               ('Input self-interaction', r's_i\had(c_i+B_is_i)+s_i'),
               (r'No input skip', r's_i\had c_i'),
               (r'Projected input skip', r's_i\had c_i+\ell_i'),
               (r'Shifted input skip', r"s_i\had c_i+s'_i"),
               (r'Channel overlap', r's_i\had c_i+s_i')]
    if len(core) != len(formats):
        raise ValueError('Core table membership and formula definitions differ')
    lines = []
    for i, (row, (label, formula)) in enumerate(zip(core, formats)):
        if i in [1, 5, 7, 10]: lines.append(r'\midrule')
        d = '--' if row['arm'] == REF else percent(row['delta_auc_percent'])
        md = '' if row['matched_delta_auc_percent'] == '' else percent(row['matched_delta_auc_percent'])
        if row['p_holm13'] != '' and row['p_holm13'] < .05: d += r'$^*$'
        if row['matched_p_holm13'] != '' and row['matched_p_holm13'] < .05: md += r'$^*$'
        lines.append(f'{label} & ${formula}$ & {row["flops_ratio"]:.2f} & {d} & {md}'.rstrip() + r' \\')
    (out / 'table3_rows.tex').write_text('\n'.join(lines) + '\n')
    t4 = []; lines = []
    for label, arm in [(None, None), (r'$1/2$ (\ours{})', REF), ('$1$ (constant)', 'width_ratio_1_matched_flops'),
                       ('$1/4$', 'width_ratio_1_4_matched_flops'), ('$1/8$', 'width_ratio_1_8_matched_flops'),
                       (None, None), ('$1$ (constant)', 'width_ratio_1'), ('$1/4$', 'width_ratio_1_4'),
                       ('$1/8$', 'width_ratio_1_8')]:
        if arm is None:
            lines.append((r'\midrule'+'\n' if lines else '') + r'\multicolumn{5}{@{}l}{\emph{' + ('Near-matched FLOPs' if not lines else 'Same first width, $d_1=W$') + r'}} \\'); continue
        rs = runs('criteo', arm); rr = runs('criteo', REF)
        delta = 100*(mean(rs)/mean(rr)-1)
        widths = rs[0]['args']['dims']
        t4.append({'arm': arm, 'widths': widths, 'matrix_flops': matrix_flops(rs[0]), 'auc': mean(rs),
                   'auc_sample_sd': st.stdev(metric(rs)), 'delta_auc_percent': delta})
        auc = mean_text(metric(rs)); auc = r'\textbf{' + auc + '}' if arm == REF else auc
        auc += r' {\scriptsize$\pm$' + f'{st.stdev(metric(rs)):.5f}'[1:] + '}'
        cells = [label, str(widths), f'{matrix_flops(rs[0])/1e6:.2f}M', auc,
                 '--' if arm == REF else percent(delta)]
        lines.append(' & '.join(cells) + r' \\')
    (out / 'table4_rows.tex').write_text('\n'.join(lines) + '\n'); write_csv(out / 'table4_schedule.csv', t4)
    (out / 'summary.json').write_text(json.dumps({'main_statistics': stats, 'seeds_by_table': {'table1': SEEDS, 'table2': SEEDS, 'table3': TABLE3_SEEDS, 'table4': SEEDS}, 'table3_statistics': table3_stats, 'records_loaded': len(records), 'table3_alternatives': len(t3)-1}, indent=2) + '\n')
    print(json.dumps({'output_dir': str(out), 'main_statistics': stats, 'records_loaded': len(records)}, indent=2))


if __name__ == '__main__':
    main()
