"""Rebuild all paper table rows and compare against the frozen manuscript rows."""
import argparse
import json
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
        pairs = [('table1_extended_matrix_gpt_rows.tex', 'table1_rows.tex'),
                 ('table2_gpt_rows.tex', 'table2_rows.tex'),
                 ('table3_core_gpt_rows.tex', 'table3_core_rows.tex'),
                 ('table3_full_gpt_rows.tex', 'table3_full_rows.tex'),
                 ('table4_gpt_rows.tex', 'table4_rows.tex')]
        for expected, actual in pairs:
            if rows(reference / 'tables' / expected) != rows(out / actual):
                raise AssertionError(f'Rows differ: {expected}')
        if json.loads((reference / 'holm36_summary.json').read_text()) != json.loads((out / 'holm36_summary.json').read_text()):
            raise AssertionError('Holm36 statistics differ')
        print('PASS: all four main tables, full 31-alternative table, and exact Welch/Holm summary match the frozen paper.')


if __name__ == '__main__':
    main()
