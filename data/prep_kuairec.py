"""KuaiRec 2.0 big matrix -> CTR CSVs (label + categorical fields), then encode with prep_bars_x1.py.
Label: watch_ratio > 2 (the KuaiRec paper's positive definition, Sec. 3.1.3: watched completely at least twice).
Fields: user_id, video_id, hour, weekday; user_features categorical columns (user_active_degree,
is_lowactive_period, is_live_streamer, is_video_author, *_range, onehot_feat0-17; raw counts dropped);
item_categories tag list (one token); static item attributes from item_daily_features (first record per video:
author_id, video_type, upload_type, music_id, video_tag_id). No play statistics (they would leak the label).
Split: uniform random 8:1:1 with numpy seed 2024.  Usage: python prep_kuairec.py <kuairec_data_dir> <out_dir>"""
import os, sys
import numpy as np
import pandas as pd

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", help="Directory containing the source CSV files")
    parser.add_argument("out", help="Directory to write generated files")
    args = parser.parse_args(argv)
    SRC, OUT = args.src, args.out
    os.makedirs(OUT, exist_ok=True)
    inter = pd.read_csv(f"{SRC}/big_matrix.csv", usecols=["user_id", "video_id", "time", "date", "watch_ratio"])
    inter = inter.dropna(subset=["time", "date"])
    t = pd.to_datetime(inter["time"], errors="coerce")
    df = pd.DataFrame({"label": (inter["watch_ratio"] > 2.0).astype(int),
                       "user_id": inter["user_id"], "video_id": inter["video_id"],
                       "hour": t.dt.hour, "weekday": pd.to_datetime(inter["date"].astype(int).astype(str), format="%Y%m%d").dt.weekday})
    uf = pd.read_csv(f"{SRC}/user_features.csv")
    ucols = ["user_active_degree", "is_lowactive_period", "is_live_streamer", "is_video_author", "follow_user_num_range",
             "fans_user_num_range", "friend_user_num_range", "register_days_range"] + [f"onehot_feat{i}" for i in range(18)]
    df = df.merge(uf[["user_id"] + ucols], on="user_id", how="left")
    ic = pd.read_csv(f"{SRC}/item_categories.csv").rename(columns={"feat": "tags"})
    df = df.merge(ic, on="video_id", how="left")
    idf = pd.read_csv(f"{SRC}/item_daily_features.csv",
                      usecols=["video_id", "date", "author_id", "video_type", "upload_type", "music_id", "video_tag_id"])
    idf = idf.sort_values(["video_id", "date"]).drop_duplicates("video_id").drop(columns="date")
    df = df.merge(idf, on="video_id", how="left")
    cols = ["label", "user_id", "video_id", "hour", "weekday"] + ucols + ["tags", "author_id", "video_type",
                                                                         "upload_type", "music_id", "video_tag_id"]
    df = df[cols]
    for c in cols[1:]:  # categorical strings; integer-valued columns without '.0'
        s = df[c]
        df[c] = s.map(lambda v: "" if pd.isna(v) else (str(int(v)) if isinstance(v, float) and float(v).is_integer() else str(v)))
    perm = np.random.default_rng(2024).permutation(len(df))
    n = len(df); a, b = int(0.8 * n), int(0.9 * n)
    for name, idx in (("train", perm[:a]), ("valid", perm[a:b]), ("test", perm[b:])):
        df.iloc[np.sort(idx)].to_csv(f"{OUT}/{name}.csv", index=False)
        print(name, len(idx), "pos_rate", round(float(df["label"].values[idx].mean()), 4), flush=True)
    print("fields", len(cols) - 1)


if __name__ == "__main__":
    main()
