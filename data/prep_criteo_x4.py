"""Criteo_x4_001 CSV-to-cache preprocessing, reconstructed from the metadata of the cache used in the experiments.

I1--I13 use floor(log(x)**2) for x > 2, otherwise x itself; numeric
zero/missing values and empty categorical values map to per-field OOV.
The logarithm is natural. This is NOT floor(log2(x)).

Vocabularies use training data only, frequency >= 10, lexicographic string
ordering, and global per-field offsets. No PAD index is allocated.
The encoder has not been verified row by row against that cache; see data/README.md.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

LABEL = "Label"
NUMERIC_FIELDS = [f"I{i}" for i in range(1, 14)]
FIELDS = NUMERIC_FIELDS + [f"C{i}" for i in range(1, 27)]


def discretize_numeric(values):
    """Convert integer-valued Criteo input to the observed categorical tokens."""
    values = pd.to_numeric(values, errors="raise").fillna(0).to_numpy(np.int64)
    positive_logs = np.floor(np.log(np.maximum(values, 1)) ** 2).astype(np.int64)
    return np.where(values > 2, positive_logs, values).astype(str)


def chunks(src, split, chunk_size):
    for frame in pd.read_csv(
        src / f"{split}.csv", dtype=object, chunksize=chunk_size,
        usecols=[LABEL] + FIELDS,
    ):
        for field in NUMERIC_FIELDS:
            frame[field] = discretize_numeric(frame[field])
        for field in FIELDS[13:]:
            frame[field] = frame[field].fillna("").astype(str)
        yield frame[LABEL].astype(float).to_numpy(np.float32), frame[FIELDS]


def check_reference(meta, reference_path):
    reference = json.loads(reference_path.read_text())
    differences = {
        key: {"computed": meta[key], "reference": reference.get(key)}
        for key in ("fields", "field_vocab", "offsets", "vocab_size")
        if meta[key] != reference.get(key)
    }
    if differences:
        raise ValueError("Reference metadata mismatch: " + json.dumps(differences))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("src", type=Path, help="Directory with the existing BARS split CSVs")
    parser.add_argument("out", type=Path, help="Directory to write the encoded NumPy cache")
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument(
        "--reference-meta", type=Path,
        help="Optional existing meta.json; verify vocabulary sizes and offsets before encoding",
    )
    args = parser.parse_args(argv)
    if args.min_count < 1 or args.chunk_size < 1:
        parser.error("--min-count and --chunk-size must be positive")
    if args.src.resolve() == args.out.resolve():
        parser.error("source and cache output directories must differ")
    for split in ("train", "valid", "test"):
        if not (args.src / f"{split}.csv").is_file():
            parser.error(f"missing source split: {split}.csv")

    counts = {field: Counter() for field in FIELDS}
    train_rows = 0
    for _, frame in chunks(args.src, "train", args.chunk_size):
        train_rows += len(frame)
        for field in FIELDS:
            counts[field].update(frame[field].value_counts().to_dict())
        print(f"counted {train_rows:,} training rows", flush=True)

    vocab, offsets, offset = {}, {}, 0
    for field in FIELDS:
        na_token = "0" if field in NUMERIC_FIELDS else ""
        kept = sorted(
            token for token, count in counts[field].items()
            if count >= args.min_count and token != na_token
        )
        vocab[field] = pd.Index(kept)
        offsets[field] = offset
        offset += len(kept) + 1
    del counts
    meta = {
        "vocab_size": offset, "fields": FIELDS, "offsets": offsets,
        "field_vocab": {field: len(vocab[field]) + 1 for field in FIELDS},
        "min_categr_count": args.min_count,
        "numeric_transform": "floor(ln(x)**2) if x > 2 else x; zero/missing -> OOV",
        "preprocessor_provenance": "reconstructed from cache metadata; not verified row by row",
    }
    if args.reference_meta:
        check_reference(meta, args.reference_meta)
    args.out.mkdir(parents=True, exist_ok=True)
    field_offsets = np.array([offsets[field] for field in FIELDS])
    for split in ("train", "valid", "test"):
        if split == "train":
            rows = train_rows
        else:
            rows = sum(len(frame) for frame in pd.read_csv(
                args.src / f"{split}.csv", usecols=[LABEL], chunksize=args.chunk_size,
            ))
        if rows == 0:
            np.save(args.out / f"{split}_cats.npy", np.empty((0, len(FIELDS)), np.int32))
            np.save(args.out / f"{split}_y.npy", np.empty(0, np.float32))
            meta[split] = {"rows": 0, "pos_rate": None, "oov_rate": None}
            continue
        cats = np.lib.format.open_memmap(
            args.out / f"{split}_cats.npy", mode="w+", dtype=np.int32,
            shape=(rows, len(FIELDS)),
        )
        labels = np.lib.format.open_memmap(
            args.out / f"{split}_y.npy", mode="w+", dtype=np.float32, shape=(rows,),
        )
        start, positive, oov_cells = 0, 0.0, 0
        for y, frame in chunks(args.src, split, args.chunk_size):
            block = np.empty((len(frame), len(FIELDS)), dtype=np.int32)
            for column, field in enumerate(FIELDS):
                block[:, column] = vocab[field].get_indexer(frame[field].to_numpy()) + 1 + offsets[field]
            end = start + len(frame)
            cats[start:end], labels[start:end] = block, y
            positive += float(y.sum(dtype=np.float64))
            oov_cells += int((block == field_offsets).sum())
            start = end
        cats.flush()
        labels.flush()
        del cats, labels
        meta[split] = {
            "rows": rows, "pos_rate": positive / rows,
            "oov_rate": oov_cells / (rows * len(FIELDS)),
        }
        print(split, meta[split], flush=True)
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")


if __name__ == "__main__":
    main()
