"""Verify table metrics, paired/Holm statistics and complete configuration coverage."""
import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def read_csv(path):
    with path.open() as f: return list(csv.DictReader(f))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=ROOT/'results')
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='pyramiddcn_tables_') as temp:
        out=Path(temp)
        subprocess.run([sys.executable,'-m','reproduce.build_tables','--results-dir',str(args.results_dir.resolve()),'--out-dir',str(out)],cwd=ROOT,check=True)
        ref=ROOT/'results/reference'
        for t in range(1,5):
            name=f'table{t}_rows.tex'
            assert (out/name).read_text()==(ref/'tables'/name).read_text(),name
        cases=[('main_comparisons.csv','p_holm32',64),('table2_significance.csv','p_holm16',16),
               ('table3_significance.csv','p_holm13',13),('table4_significance.csv','p_holm6',6)]
        for t,(filename,adjusted,n) in enumerate(cases,1):
            actual=read_csv(out/filename)
            expected=[r for r in read_csv(ref/'manuscript_statistics.csv') if int(r['table'])==t]
            assert len(actual)==len(expected)==n
            def comparison_key(row):
                if t == 1:
                    model = 'fcn_p' if row['baseline'] == 'fcn' else row['baseline']
                    return (f"{model} / {row['dataset']}", row['metric'])
                if t == 2:
                    model = 'fcn_p' if row['model'] == 'fcn' else row['model']
                    condition = {'pyramid': 'same_input', 'pyramid_matched_flops': 'matched'}[row['configuration']]
                    return (f"{model} / {condition}", 'test_auc')
                if t == 3:
                    names = {'input_pooling': 'Input pooling', 'frozen_projection': 'Frozen projection',
                             'learned_projection': 'Learned projection', 'previous_layer_channels': 'Previous-layer channels',
                             'addition_only': 'Addition only', 'input_self_interaction': 'Input self-interaction',
                             'no_input_skip': 'No input skip', 'projected_input_skip': 'Projected input skip',
                             'shifted_input_skip': 'Shifted input skip', 'channel_overlap': 'Channel overlap'}
                    return (f"{names[row['core_arm']]} / {row['comparison']}", 'test_auc')
                arm = row['arm']
                condition = 'Matched FLOPs' if arm.endswith('_matched_flops') else 'Same first width'
                ratio = arm.removeprefix('width_ratio_').removesuffix('_matched_flops')
                return (f"{condition}: " + {'1': 'constant', '1_4': 'quarter', '1_8': 'eighth'}[ratio], 'test_auc')
            expected_by_key = {(r['comparison'], r['metric']): r for r in expected}
            assert len(expected_by_key) == n
            assert {comparison_key(a) for a in actual} == set(expected_by_key), (t, {comparison_key(a) for a in actual} ^ set(expected_by_key))
            for a in actual:
                b = expected_by_key[comparison_key(a)]
                for x,y in [('t','t'),('p_raw','p_raw'),(adjusted,'p_holm')]:
                    assert math.isclose(float(a[x]),float(b[y]),rel_tol=1e-7,abs_tol=1e-10),(t,x)
        index=json.loads((ROOT/'configs/experiment_index.json').read_text())
        allnames=set()
        for table,datasets in index['groups'].items():
            expected_seeds=set(index['seeds_by_group'][table])
            for ds,names in datasets.items():
                configs=json.loads((ROOT/f'configs/{table}/{ds}.json').read_text())
                assert len(names)==len(set(names))==len(configs)
                assert {r['name'] for r in configs}==set(names)
                allnames.update(names)
                records=[json.loads(line) for line in (args.results_dir/f'{ds}.jsonl').read_text().splitlines() if line.strip()]
                assert len(records)==len({r['name'] for r in records}),'duplicate run names'
                lookup={r['name']:r for r in records}
                cohorts={}
                for c in configs:
                    assert c['max_epochs']==1
                    r=lookup[c['name']]
                    assert all(r.get(k)==v for k,v in c.items()),f'Config mismatch: {c["name"]}'
                    assert all(math.isfinite(r[k]) for k in ['test_auc','test_logloss','linear_flops'])
                    cohorts.setdefault(c['name'].rsplit('_s',1)[0],[]).append(c['seed'])
                assert all(len(v)==len(expected_seeds) and set(v)==expected_seeds for v in cohorts.values())
        assert len(allnames)==index['unique_runs']==610
        for ds in index['datasets']:
            selected = set().union(*(set(group.get(ds, [])) for group in index['groups'].values()))
            catalog = json.loads((ROOT/f'configs/all/{ds}.json').read_text())
            records = [json.loads(line) for line in (args.results_dir/f'{ds}.jsonl').read_text().splitlines() if line.strip()]
            assert len(catalog) == len(selected) == len(records), f'Extra or missing runs: {ds}'
            assert {r['name'] for r in catalog} == {r['name'] for r in records} == selected
        print('PASS: 610 selected runs, all four tables and all 99 paired/Holm comparisons.')

if __name__=='__main__':main()
