"""BARS KKBox_x1, and the KuaiRec CSVs written by prep_kuairec.py -> the cached format of prep_avazu_x4.py.

Every column except the label is read as a categorical string (na -> "", encoded as OOV), including
KKBox's multi-valued fields (genre_ids, composer, lyricist, ...), which are kept as single tokens rather
than FuxiCTR sequence features. Vocab per field from train only, count >= MIN_COUNT; index 0 = OOV of
that field; fields share one table via global offsets in CSV column order.
Writes <out>/{split}_cats.npy (int32, N x F), {split}_y.npy (float32), meta.json.
Usage: python prep_bars_x1.py <src_dir> <out_dir> <min_count>
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", help="Directory containing the source CSV files")
    parser.add_argument("out", help="Directory to write generated files")
    parser.add_argument("min_count", type=int, help="Minimum training-set category frequency")
    args = parser.parse_args(argv)
    SRC, OUT, MIN_COUNT = args.src, args.out, args.min_count
    if MIN_COUNT < 1:
        parser.error("min_count must be positive")
    LABEL, CHUNK, NA = "label", 1_000_000, ""
    FIELDS = [c for c in pd.read_csv(f"{SRC}/train.csv", nrows=1).columns if c != LABEL]


    def chunks(split, nan_counts):
        for df in pd.read_csv(f"{SRC}/{split}.csv", dtype=object, chunksize=CHUNK, keep_default_na=True):
            for c, n in df.isnull().sum().items():
                nan_counts[c] = nan_counts.get(c, 0) + int(n)
            for c in FIELDS:
                df[c] = df[c].fillna(NA).astype(str)
            yield df[LABEL].astype(float).to_numpy(np.float32), df[FIELDS]


    os.makedirs(OUT, exist_ok=True)
    counts = {f: Counter() for f in FIELDS}
    for _, df in chunks("train", {}):
        for f in FIELDS:
            counts[f].update(df[f].value_counts().to_dict())
    vocab, offsets, offset = {}, {}, 0
    for f in FIELDS:
        kept = sorted(k for k, n in counts[f].items() if n >= MIN_COUNT and k != NA)
        vocab[f] = pd.Index(kept)
        offsets[f] = offset
        offset += len(kept) + 1  # +1: per-field OOV at local index 0
    print(f"{len(FIELDS)} fields, total vocab {offset:,}", flush=True)
    meta = {"vocab_size": offset, "fields": FIELDS, "offsets": offsets,
            "field_vocab": {f: len(vocab[f]) + 1 for f in FIELDS}, "min_categr_count": MIN_COUNT,
            "source": os.path.abspath(SRC)}
    for split in ("train", "valid", "test"):
        ys, cats, nan_counts = [], [], {}
        for y, df in chunks(split, nan_counts):
            block = np.empty((len(df), len(FIELDS)), dtype=np.int32)
            for j, f in enumerate(FIELDS):
                block[:, j] = vocab[f].get_indexer(df[f].to_numpy()) + 1 + offsets[f]  # -1 (OOV) -> 0
            ys.append(y)
            cats.append(block)
        y, c = np.concatenate(ys), np.concatenate(cats)
        np.save(f"{OUT}/{split}_y.npy", y)
        np.save(f"{OUT}/{split}_cats.npy", c)
        oov = float((c == np.array([offsets[f] for f in FIELDS])).mean())
        meta[split] = {"rows": int(len(y)), "pos_rate": float(y.mean()), "oov_rate": oov,
                       "nan_cells": int(sum(nan_counts.values()))}
        print(split, meta[split], flush=True)
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
