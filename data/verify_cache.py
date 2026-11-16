"""Compare prepared data with the selected experiments without exposing examples."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def verify(dataset, directory):
    expected = json.loads((ROOT / 'configs/cache_fingerprints.json').read_text())[dataset]
    metadata = json.loads((directory / 'meta.json').read_text())
    for key, value in expected['metadata'].items():
        if metadata.get(key) != value:
            raise ValueError(f'{dataset}: metadata mismatch for {key}')
    for filename, spec in expected['arrays'].items():
        array = np.load(directory / filename, mmap_mode='r', allow_pickle=False)
        if list(array.shape) != spec['shape'] or str(array.dtype) != spec['dtype']:
            raise ValueError(f'{filename}: shape/dtype mismatch')
        rows = np.linspace(0, len(array) - 1, spec['sample_rows'], dtype=np.int64)
        digest = hashlib.sha256(np.ascontiguousarray(array[rows]).tobytes()).hexdigest()
        if digest != spec['sample_sha256']:
            raise ValueError(f'{filename}: sampled values differ from the experimental cache')
    print(f'PASS {dataset}: metadata, six array shapes/dtypes and sampled fingerprints. '
          'This checks 1,024 evenly spaced rows per array, not every row.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['criteo', 'avazu', 'kkbox', 'kuairec'])
    parser.add_argument('--data-dir', required=True, type=Path)
    args = parser.parse_args()
    verify(args.dataset, args.data_dir)


if __name__ == '__main__':
    main()
