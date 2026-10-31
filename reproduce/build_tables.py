"""Recompute Tables 1–4 and exact Welch/Holm statistics from scalar run records.

Uses the frozen published records by default, or JSONL records in --results-dir.
No training, data downloads, manuscript parsing, or fixed-t significance shortcuts.
"""
import argparse
import csv
import json
import math
import statistics as st
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from scipy.stats import ttest_ind
from .matrix_flops import matrix_flops

ROOT = Path(__file__).resolve().parents[1]
DS = ['criteo', 'avazu', 'kkbox', 'kuairec']
SEEDS = [2024, 2025, 2026]
REF = 'v3n_stagger'
BASE = [('dcnv2_lr', r'DCNv2, $r=\lfloor W/4\rfloor$~\cite{wang2021dcnv2}'),
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
    def runs(ds, arm, seeds=SEEDS, baseline=False):
        prefix = f'v3_{ds}_{arm}' if baseline else f'fair_{ds}_{arm}'
        return [records[f'{prefix}_s{s}'] for s in seeds]
    def metric(rs, key='test_auc'):
        return [r[key] for r in rs]
    def mean(rs):
        return st.mean(metric(rs))
    def table1_runs(ds, k):
        return runs(ds, 'v3f_r4') if k == 'dcnv2_lr' else runs(ds, k, baseline=True)
    comparisons = []
    for ds in DS:
        for key, _ in BASE:
            for m in ['test_auc', 'test_logloss']:
                x, y = metric(runs(ds, REF), m), metric(table1_runs(ds, key), m)
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
            rs = runs(ds, REF) if k == 'ours' else table1_runs(ds, k)
            for m, sign in [('test_auc', 1), ('test_logloss', -1)]:
                values = metric(rs, m)
                best = max([sign * st.mean(metric(table1_runs(ds, b), m)) for b, _ in BASE] + [sign * st.mean(metric(runs(ds, REF), m))])
                number = mean_text(values)
                if sign * st.mean(values) >= best - 1e-12: number = r'\mathbf{' + number + '}'
                cells.append('$' + number + r'\,{\scriptstyle\pm' + f'{st.stdev(values):.5f}'[1:] + '}$')
            cells.append(f'{matrix_flops(rs[0])/1e6:.2f}M')
        if k == 'ours': t1.append(r'\midrule')
        t1.append(label + ' & ' + ' & '.join(cells) + r' \\')
    (out / 'table1_rows.tex').write_text('\n'.join(t1) + '\n')
    t2 = []; t2tests = []
    for k, label in BASE:
        original = runs('criteo', k, baseline=True)
        variants = []
        for typ in ['pyr', 'eqa']:
            arm = {'dcnv2_lr': f'v3p_{typ}2_dcnv2', 'qnn': f'v3p_{typ}2_qnn'}.get(k, f'v3p_{typ}_{k}')
            rs = runs('criteo', arm); test = ttest_ind(metric(rs), metric(original), equal_var=False)
            t2tests.append({'model': k, 'configuration': typ, 'delta_auc': mean(rs)-mean(original),
                            't': float(test.statistic), 'df': float(test.df), 'p_raw': float(test.pvalue)})
            variants.append((rs, float(test.pvalue)))
        p, q = [r for r, _ in variants]
        star = lambda index: r'$^*$' if variants[index][1] < .05 else ''
        short = 'DCNv2' if k == 'dcnv2_lr' else label.split(r'~\cite')[0]
        cells = [short, mean_text(metric(original)), f'{matrix_flops(original[0])/1e6:.2f}',
                 mean_text(metric(p)) + star(0), percent(100*(mean(p)/mean(original)-1)),
                 f'{matrix_flops(p[0])/1e6:.2f}', percent(100*(matrix_flops(p[0])/matrix_flops(original[0])-1), 0),
                 mean_text(metric(q)) + star(1), percent(100*(mean(q)/mean(original)-1))]
        t2.append(' & '.join(cells) + r' \\')
    ref = runs('criteo', REF)
    t2 += [r'\midrule', r'\ours{} & \textbf{' + mean_text(metric(ref)) + r'} & \textbf{' + f'{matrix_flops(ref[0])/1e6:.2f}' + r'} & & & & & & \\']
    (out / 'table2_rows.tex').write_text('\n'.join(t2) + '\n')
    for typ in ['pyr', 'eqa']: holm([r for r in t2tests if r['configuration'] == typ], 'p_holm9')
    write_csv(out / 'table2_welch.csv', t2tests)
    specs = json.loads((ROOT / 'configs' / 'table3_specification.json').read_text())
    t3 = []
    for spec in specs:
        arm = spec['arm']
        seeds = [s for s in range(2024, 2032) if all(f'fair_{ds}_{arm}_s{s}' in records and f'fair_{ds}_{REF}_s{s}' in records for ds in DS)]
        def aggregate(a, shared):
            aa = {ds: runs(ds, a, shared) for ds in DS}; bb = {ds: runs(ds, REF, shared) for ds in DS}
            # Normalize by each dataset's mean reference, then average over the four fixed datasets.
            delta = 100 * st.mean(mean(aa[ds]) / mean(bb[ds]) - 1 for ds in DS)
            cost = st.mean(matrix_flops(aa[ds][0])/matrix_flops(bb[ds][0]) for ds in DS)
            return delta, cost
        delta, cost = aggregate(arm, seeds)
        matched = spec['matched_arm']; matched_delta = ''
        if matched:
            shared = [s for s in range(2024, 2032) if all(f'fair_{ds}_{matched}_s{s}' in records and f'fair_{ds}_{REF}_s{s}' in records for ds in DS)]
            matched_delta = aggregate(matched, shared)[0]
        t3.append(dict(spec, seeds=seeds, flops_ratio=round(cost, 2), delta_auc_percent=delta, matched_delta_auc_percent=matched_delta))
    write_csv(out / 'table3_full.csv', t3)
    core_arms = json.loads((ROOT / 'configs' / 'experiment_index.json').read_text())['core_table3_arms']
    core = [next(r for r in t3 if r['arm'] == a) for a in core_arms]
    write_csv(out / 'table3_core.csv', core)
    # Formula columns use the explicit published definitions; numeric cells are all recomputed above.
    formats = [(r'\textbf{\ours{}}', r'x[S_i]\had(A_ix_{i-1}+b_i)+x[S_i]'),
               ('Learned input map', r'L_ix\had(A_ix_{i-1}+b_i)+L_ix'),
               (r'Previous-state slice$^\dagger$', r'x_{i-1}[T_i]\had(A_ix_{i-1}+b_i)+x_{i-1}[T_i]'),
               ('No product', r'x[S_i]+A_ix_{i-1}+b_i'),
               (r'No input addend$^\dagger$', r'x[S_i]\had(A_ix_{i-1}+b_i)'),
               ('Extra input cross', r'x[S_i]\had(A_ix_{i-1}+B_ix[S_i]+b_i)+x[S_i]'),
               (r'Nested slices$^\dagger$ ($S_3\subset S_2$)', r'x[S_i]\had(A_ix_{i-1}+b_i)+x[S_i]')]
    lines = []
    for i, (row, (label, formula)) in enumerate(zip(core, formats)):
        if i in [1, 3, 6]: lines.append(r'\midrule')
        d = '--' if row['arm'] == REF else percent(row['delta_auc_percent'])
        md = '' if row['matched_delta_auc_percent'] == '' else percent(row['matched_delta_auc_percent'])
        lines.append(f'{label} & ${formula}$ & {row["flops_ratio"]:.2f} & {d} & {md}' + r' \\')
    (out / 'table3_core_rows.tex').write_text('\n'.join(lines) + '\n')
    lines = []
    for row in t3:
        d = '--' if row['arm'] == REF else percent(row['delta_auc_percent'])
        md = '' if row['matched_delta_auc_percent'] == '' else percent(row['matched_delta_auc_percent'])
        lines.append(f'{row["label"]} & {row["formula"]} & {row["flops_ratio"]:.2f} & {d} & {md}' + r' \\')
    (out / 'table3_full_rows.tex').write_text('\n'.join(lines) + '\n')
    t4 = []; lines = []
    for label, arm in [(None, None), (r'$1/2$ (\ours{})', REF), ('$1$ (constant)', 'v3n_o1_bal'),
                       ('$1/4$', 'v3n_o4_bal'), ('$1/8$', 'v3n_o8_bal'),
                       (None, None), ('$1$ (constant)', 'v3n_r1'), ('$1/4$', 'v3n_r4_bal'), ('$1/8$', 'v3n_r8_bal')]:
        if arm is None:
            lines.append((r'\midrule'+'\n' if lines else '') + r'\multicolumn{7}{@{}l}{\emph{' + ('Matched FLOPs' if not lines else 'Matched first width, $d_1=W$') + r'}} \\'); continue
        rs = runs('criteo', arm); rr = runs('criteo', REF)
        ev = [r['erank']['rep_frac'][1:] for r in rs if r.get('erank')]
        if not ev: ev = [records[f'erk_criteo_{arm}_s{s}']['erank']['rep_frac'][1:] for s in SEEDS]
        per = [100*st.mean(v[i] for v in ev) for i in range(3)]
        delta = 100*(mean(rs)/mean(rr)-1)
        t4.append({'arm': arm, 'widths': rs[0]['dims'], 'matrix_flops': matrix_flops(rs[0]), 'auc': mean(rs),
                   'delta_auc_percent': delta, 'effective_rank_percent': per, 'mean_effective_rank_percent': st.mean(per)})
        auc = mean_text(metric(rs)); auc = r'\textbf{' + auc + '}' if arm == REF else auc
        cells = [label, str(rs[0]['dims']), f'{matrix_flops(rs[0])/1e6:.2f}M', auc,
                 '--' if arm == REF else percent(delta), '/'.join(str(round(v)) for v in per), str(round(st.mean(per)))]
        lines.append(' & '.join(cells) + r' \\')
    (out / 'table4_rows.tex').write_text('\n'.join(lines) + '\n'); write_csv(out / 'table4_schedule.csv', t4)
    no_ln = st.mean(mean(runs(ds, 'v3n_noln_dj')) - mean(runs(ds, REF)) for ds in DS)
    (out / 'summary.json').write_text(json.dumps({'main_statistics': stats, 'no_ln_mean_absolute_auc_delta': no_ln,
                                                 'records_loaded': len(records), 'table3_alternatives': len(t3)-1}, indent=2) + '\n')
    print(json.dumps({'output_dir': str(out), 'main_statistics': stats, 'records_loaded': len(records)}, indent=2))


if __name__ == '__main__':
    main()
