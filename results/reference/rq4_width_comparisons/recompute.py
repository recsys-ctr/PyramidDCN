"""Recompute the RQ4 width comparisons from the saved runs.

    python results/reference/rq4_width_comparisons/recompute.py          # check summary.json and per_seed.csv
    python results/reference/rq4_width_comparisons/recompute.py --write  # regenerate them
"""
import argparse
import csv
import io
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
DATASETS = ['criteo', 'avazu', 'kkbox', 'kuairec']
SEEDS = [2024, 2025, 2026]
SCHEDULES = [('default', 'pyramiddcn'), ('constant', 'width_ratio_1_matched_flops'),
             ('widening', 'width_ratio_2_matched_flops')]
EMB_DIM = 16
METRICS = [('test_auc', 'test_AUC'), ('test_logloss', 'test_LogLoss')]


def stats(values):
    return {'mean': st.mean(values), 'sample_SD': st.stdev(values)}


def compute():
    metadata = json.loads((REPO / 'configs' / 'dataset_metadata.json').read_text())
    records = {}
    for line in (ROOT / 'runs.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r['name'] in records:
            raise ValueError(f'Duplicate run name: {r["name"]}')
        if r['max_epochs'] != 1 or r['batch'] != 4096:
            raise ValueError(f'Unexpected protocol in {r["name"]}')
        records[r['name']] = r
    summary = {'delta_convention': 'default [W, W/2, W/4] minus comparison; positive AUC and negative LogLoss favor the default.',
               'datasets': {}}
    rows = []
    for ds in DATASETS:
        embedding_width = len(metadata[ds]['fields']) * EMB_DIM
        groups = {}
        for label, arm in SCHEDULES:
            runs = [records.get(f'{ds}_{arm}_s{seed}') for seed in SEEDS]
            if all(r is None for r in runs):
                continue
            if any(r is None for r in runs):
                raise ValueError(f'Incomplete seed cohort: {ds} {arm}')
            width, dims = runs[0]['input_proj'], runs[0]['args']['dims']
            # Eq. (2): matrix FLOPs = 2 (D W + sum_i d_{i-1} d_i + d_L), with d_0 = W.
            macs = embedding_width * width + sum(a * b for a, b in zip([width] + dims[:-1], dims)) + dims[-1]
            for r in runs:
                if r['args']['dims'] != dims or r['input_proj'] != width or r['linear_macs'] != macs or r['linear_flops'] != 2 * macs:
                    raise ValueError(f'Inconsistent widths or FLOPs in {r["name"]}')
            groups[label] = {'widths': dims, 'projected_input_width': width, 'matrix_FLOPs': 2 * macs,
                             'metrics': {out: stats([r[key] for r in runs]) for key, out in METRICS},
                             'run_names': [r['name'] for r in runs]}
        default = [records[name] for name in groups['default']['run_names']]
        for label, group in groups.items():
            group['FLOPs_ratio_to_default'] = group['matrix_FLOPs'] / groups['default']['matrix_FLOPs']
            if label == 'default':
                continue
            controls = [records[name] for name in group['run_names']]
            group['paired_differences'] = {
                out: dict(stats([x[key] - y[key] for x, y in zip(default, controls)]),
                          per_seed=[{'seed': seed, 'difference': x[key] - y[key]} for seed, x, y in zip(SEEDS, default, controls)])
                for key, out in METRICS}
            for seed, x, y in zip(SEEDS, default, controls):
                rows.append({'dataset': ds, 'schedule': label, 'seed': seed,
                             'default_run': x['name'], 'comparison_run': y['name'],
                             'default_widths': str(x['args']['dims']), 'comparison_widths': str(y['args']['dims']),
                             'default_test_AUC': x['test_auc'], 'comparison_test_AUC': y['test_auc'],
                             'difference_test_AUC': x['test_auc'] - y['test_auc'],
                             'default_test_LogLoss': x['test_logloss'], 'comparison_test_LogLoss': y['test_logloss'],
                             'difference_test_LogLoss': x['test_logloss'] - y['test_logloss'],
                             'default_matrix_FLOPs': x['linear_flops'], 'comparison_matrix_FLOPs': y['linear_flops']})
        summary['datasets'][ds] = groups
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return json.dumps(summary, indent=2) + '\n', buffer.getvalue(), len(records), len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--write', action='store_true', help='Regenerate summary.json and per_seed.csv')
    args = parser.parse_args()
    summary, per_seed, runs, pairs = compute()
    outputs = {'summary.json': summary, 'per_seed.csv': per_seed}
    for name, text in outputs.items():
        if args.write:
            (ROOT / name).write_text(text)
        elif (ROOT / name).read_text() != text:
            raise AssertionError(f'{name} differs from the recomputed values')
    print(json.dumps({'status': 'PASS', 'runs': runs, 'paired_rows': pairs, 'written': bool(args.write)}))


if __name__ == '__main__':
    main()
