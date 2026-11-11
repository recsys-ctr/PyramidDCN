"""Verify matrix FLOPs with CPU synthetic forwards, retaining matrix shapes without data."""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode
import train
from .matrix_flops import matrix_flops

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=['main', 'all'], default='main', help='main: Tables 1 and 2; all: every exported model shape')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs' / 'matrix_flops.csv')
    args = parser.parse_args()
    metadata = json.loads((ROOT / 'configs' / 'dataset_metadata.json').read_text())
    index = json.loads((ROOT / 'configs' / 'experiment_index.json').read_text())
    torch.set_num_threads(1)
    profiles = []
    for ds in index['datasets']:
        records = {r['name']: r for r in map(json.loads, (ROOT / 'results' / f'{ds}.jsonl').read_text().splitlines())}
        configs = {c['name']: c for c in json.loads((ROOT / 'configs' / 'all' / f'{ds}.json').read_text())}
        groups = ['table1', 'table2'] if args.scope == 'main' else list(index['groups'])
        names = list(dict.fromkeys(n for g in groups for n in index['groups'].get(g, {}).get(ds, [])))
        seen = set()
        for name in names:
            config = configs[name]
            signature = json.dumps({k: v for k, v in config.items() if k not in ['name', 'seed']}, sort_keys=True)
            if signature in seen:
                continue
            seen.add(signature)
            train.configure_metadata(dict(metadata[ds], vocab_size=16), dataset=f'synthetic_{ds}')
            cats = torch.zeros(1, len(metadata[ds]['fields']), dtype=torch.long)
            torch.manual_seed(0)
            model = train.CTRModel(config).cpu().eval()
            linear = 2 * train.linear_macs(model, torch.zeros(1, 0), cats)
            with torch.no_grad(), FlopCounterMode(display=False) as counter:
                prediction = model(None, cats)
            matrix = counter.get_total_flops()
            logged = records[name]['linear_flops']
            expected = matrix_flops(records[name])
            if prediction.shape != (1,) or linear != logged or matrix != expected:
                raise AssertionError(dict(name=name, shape=list(prediction.shape), logged=logged,
                                          linear_profile=linear, matrix_profile=matrix, analytic_matrix=expected))
            profiles.append({'dataset': ds, 'run_name': name, 'linear_flops': linear,
                             'matrix_flops': matrix, 'extra_matrix_flops': matrix-linear,
                             'analytic_matrix_flops': expected, 'torch_version': torch.__version__})
            print(ds, name, matrix, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(profiles[0])); writer.writeheader(); writer.writerows(profiles)
    print(f'Verified {len(profiles)} architecture/dataset configurations; {args.out}')


if __name__ == '__main__':
    main()
