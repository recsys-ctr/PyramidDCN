"""Rebuild all paper table rows and compare against the frozen manuscript rows."""
import argparse
import json
import csv
import math
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [s.strip() for s in path.read_text().splitlines() if '&' in s and not s.startswith(('%', r'\multicolumn'))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='pyramiddcn_tables_') as temp:
        out = Path(temp)
        subprocess.run([sys.executable, '-m', 'reproduce.build_tables', '--results-dir', str(args.results_dir.resolve()), '--out-dir', str(out)], check=True, cwd=ROOT)
        reference = ROOT / 'results' / 'reference'
        pairs = [('table1_rows.tex', 'table1_rows.tex'),
                 ('table2_rows.tex', 'table2_rows.tex'),
                 ('table3_rows.tex', 'table3_rows.tex'),
                 ('table4_rows.tex', 'table4_rows.tex')]
        for expected, actual in pairs:
            if rows(reference / 'tables' / expected) != rows(out / actual):
                raise AssertionError(f'Rows differ: {expected}')
        if json.loads((reference / 'holm36_summary.json').read_text()) != json.loads((out / 'holm36_summary.json').read_text()):
            raise AssertionError('Holm36 statistics differ')
        expected = list(csv.DictReader((reference / 'table3_significance.csv').open()))
        actual = list(csv.DictReader((out / 'table3_significance.csv').open()))
        if len(actual) != 13 or len(expected) != 13:
            raise AssertionError('Table 3 must contain all 13 significance comparisons')
        for a, b in zip(actual, expected):
            for key in ['arm', 'comparison', 'n_seed_blocks', 'significant_holm05']:
                if a[key] != b[key]: raise AssertionError(f'Table 3 mismatch: {key}')
            for key in ['mean_relative_AUC_difference_percent', 't', 'p_raw', 'p_holm13']:
                if not math.isclose(float(a[key]), float(b[key]), rel_tol=1e-7, abs_tol=1e-12):
                    raise AssertionError(f'Table 3 numeric mismatch: {key}')
        index = json.loads((ROOT / 'configs/experiment_index.json').read_text())
        for group, datasets in index['groups'].items():
            seeds = set(index['seeds_by_group'][group])
            for ds, names in datasets.items():
                configs = json.loads((ROOT / 'configs' / group / f'{ds}.json').read_text())
                if {r['name'] for r in configs} != set(names):
                    raise AssertionError('Run launcher configurations disagree with the index')
                cohorts = {}
                for r in configs:
                    cohorts.setdefault(r['name'].rsplit('_s', 1)[0], []).append(r['seed'])
                if any(len(v) != len(seeds) or set(v) != seeds for v in cohorts.values()):
                    raise AssertionError(f'Incomplete seed cohort in {group}/{ds}')
        print('PASS: Tables 1/2/4 (three seeds), Table 3 (eight seeds), and all significance results match.')


if __name__ == '__main__':
    main()
