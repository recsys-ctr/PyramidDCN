"""Avazu_x4_001 preprocessing, equivalent to FuxiCTR's BARS config (avazu_x4_001_a31210da).

'id' dropped. All 24 fields categorical (read as str). From the raw 'hour' string YYMMDDHH,
exactly as fuxictr/datasets/avazu.py (FuxiCTR v2.2.0; preprocess runs over feature_cols in
reverse, so weekday/weekend see the raw string before hour is converted):
  hour    = str(int(x[6:8]))
  weekday = str(int(date(int('20'+x[0:2]), int(x[2:4]), int(x[4:6])).strftime('%w')))
  weekend = '1' if %w in ['6', '0'] else '0'
Other fields: na -> "". Vocab per field from train only, min_categr_count 2; index 0 = OOV of
that field. As in FuxiCTR's Tokenizer, the na token "" is excluded from the vocab and encodes
as OOV. Fields share one table via global offsets; field order = FuxiCTR feature_cols order.
Writes <out>/{split}_cats.npy (int32, N x 24), {split}_y.npy (float32), meta.json.
"""
import json
import os
import sys
from collections import Counter
from datetime import date

import numpy as np
import pandas as pd

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", help="Directory containing the source CSV files")
    parser.add_argument("out", help="Directory to write generated files")
    parser.add_argument("reference_vocab", nargs="?", help="Optional field-to-FuxiCTR-vocabulary-size JSON")
    args = parser.parse_args(argv)
    SRC, OUT = args.src, args.out
    LABEL = "click"
    RAW_COLS = ["C1", "banner_pos", "site_id", "site_domain", "site_category", "app_id", "app_domain",
                "app_category", "device_id", "device_ip", "device_model", "device_type", "device_conn_type",
                "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]
    FIELDS = ["hour"] + RAW_COLS + ["weekday", "weekend"]
    CHUNK = 1_000_000
    MIN_COUNT = 2
    NA = ""


    # FuxiCTR v2.2.0 fuxictr/datasets/avazu.py, applied per unique raw hour string
    def _convert_weekday(timestamp):
        dt = date(int('20' + timestamp[0:2]), int(timestamp[2:4]), int(timestamp[4:6]))
        return int(dt.strftime('%w'))


    def _convert_weekend(timestamp):
        dt = date(int('20' + timestamp[0:2]), int(timestamp[2:4]), int(timestamp[4:6]))
        return 1 if dt.strftime('%w') in ['6', '0'] else 0


    def _convert_hour(x):
        return int(x[6:8])


    def chunks(split, nan_counts):
        usecols = [LABEL, "hour"] + RAW_COLS
        for df in pd.read_csv(f"{SRC}/{split}.csv", usecols=usecols, dtype=object, chunksize=CHUNK):
            for c, n in df.isnull().sum().items():
                nan_counts[c] = nan_counts.get(c, 0) + int(n)
            assert not df["hour"].isnull().any()
            raw = df["hour"].astype(str)
            uniq = pd.unique(raw)
            df["weekend"] = raw.map({u: str(_convert_weekend(u)) for u in uniq})
            df["weekday"] = raw.map({u: str(_convert_weekday(u)) for u in uniq})
            df["hour"] = raw.map({u: str(_convert_hour(u)) for u in uniq})
            for c in RAW_COLS:
                df[c] = df[c].fillna(NA).astype(str)
            yield df[LABEL].astype(float).to_numpy(np.float32), df[FIELDS]


    os.makedirs(OUT, exist_ok=True)
    counts = {f: Counter() for f in FIELDS}
    n_train = 0
    for _, df in chunks("train", {}):
        n_train += len(df)
        for f in FIELDS:
            counts[f].update(df[f].value_counts().to_dict())
        print(f"counted {n_train:,}", flush=True)

    vocab, offsets, offset = {}, {}, 0
    for f in FIELDS:
        kept = sorted(k for k, n in counts[f].items() if n >= MIN_COUNT and k != NA)
        vocab[f] = pd.Index(kept)
        offsets[f] = offset
        offset += len(kept) + 1  # +1: per-field OOV at local index 0
    print(f"total vocab {offset:,}", flush=True)

    meta = {"vocab_size": offset, "fields": FIELDS, "offsets": offsets,
            "field_vocab": {f: len(vocab[f]) + 1 for f in FIELDS}, "min_categr_count": MIN_COUNT}
    for split in ("train", "valid", "test"):
        ys, cats, nan_counts = [], [], {}
        for y, df in chunks(split, nan_counts):
            block = np.empty((len(df), len(FIELDS)), dtype=np.int32)
            for j, f in enumerate(FIELDS):
                local = vocab[f].get_indexer(df[f].to_numpy()) + 1  # -1 (OOV) -> 0
                block[:, j] = local + offsets[f]
            ys.append(y)
            cats.append(block)
        y, c = np.concatenate(ys), np.concatenate(cats)
        del ys, cats
        np.save(f"{OUT}/{split}_y.npy", y)
        np.save(f"{OUT}/{split}_cats.npy", c)
        oov = float((c == np.array([offsets[f] for f in FIELDS])).mean())
        meta[split] = {"rows": int(len(y)), "pos_rate": float(y.mean()), "oov_rate": oov,
                       "pos": int(y.sum()), "nan_cells": int(sum(nan_counts.values()))}
        print(split, meta[split], flush=True)
        del y, c
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
    # FuxiCTR vocab_size = kept + PAD + OOV; ours = kept + OOV
    ref = json.load(open(args.reference_vocab)) if args.reference_vocab else {}
    mismatch = {f: (meta["field_vocab"][f], ref[f] - 1) for f in ref if meta["field_vocab"][f] != ref[f] - 1}
    print("vocab mismatches vs FuxiCTR:", mismatch, "| fields compared:", len(ref), flush=True)


if __name__ == "__main__":
    main()
