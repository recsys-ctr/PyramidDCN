"""Run a frozen experiment group using an explicitly prepared dataset."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', '--group', dest='group', required=True, choices=['table1', 'table2', 'table3', 'table4', 'all'])
    parser.add_argument('--dataset', required=True, choices=['criteo', 'avazu', 'kkbox', 'kuairec'])
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--device', default='cpu', help='torch device, e.g. cpu, mps, cuda')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, help='Run only one seed from the frozen group')
    parser.add_argument('--name', help='Run only this run name')
    parser.add_argument('--output', '--out', dest='out', type=Path, help='Append-only result JSONL; existing run names are skipped')
    parser.add_argument('--limit', type=int, help='Run at most this many pending configurations')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    config = ROOT / 'configs' / args.group / f'{args.dataset}.json'
    if not config.exists():
        parser.error(f'{args.group} is not defined for {args.dataset}')
    arms = json.loads(config.read_text())
    arms = [a for a in arms if (args.seed is None or a['seed'] == args.seed) and (args.name is None or a['name'] == args.name)]
    out = args.out or ROOT / 'outputs' / f'{args.dataset}.jsonl'
    out = out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)['name'] for line in out.read_text().splitlines() if line.strip()} if out.exists() else set()
    pending = [a for a in arms if a['name'] not in done]
    if args.limit is not None:
        pending = pending[:args.limit]
    if not arms:
        parser.error('No configurations match the requested filters')
    if not pending:
        print('All selected runs already exist in', out)
        return
    # One child per arm keeps argv short and makes a failed run visible to the caller.
    for arm in pending:
        subprocess.run([sys.executable, '-m', 'train', '--arms', json.dumps([arm]),
                        '--data-dir', str(args.data_dir.expanduser().resolve()), '--device', args.device,
                        '--threads', str(args.threads), '--eval_test', '--out', str(out)], cwd=ROOT, check=True)
        records = {json.loads(line)['name'] for line in out.read_text().splitlines() if line.strip()} if out.exists() else set()
        if arm['name'] not in records:
            raise RuntimeError(f'Runner did not produce a result for {arm["name"]}; inspect its failure output')
    print(f'Completed {len(pending)} runs; results: {out}')


if __name__ == '__main__':
    main()
