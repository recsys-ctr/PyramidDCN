# RQ4 width comparisons

RQ4 compares the default `[W, W/2, W/4]` schedule with constant-width and
widening schedules at near-matched matrix FLOPs. This folder contains the 33
one-epoch runs behind those comparisons (seeds 2024–2026) and a script that
recomputes their summaries.

```sh
python results/reference/rq4_width_comparisons/recompute.py
```

The script uses only the Python standard library. Without arguments it checks
`summary.json` and `per_seed.csv` against `runs.jsonl`; `--write` regenerates
them.

## Runs

| Schedule | Run names | Datasets |
| --- | --- | --- |
| Default `[W, W/2, W/4]` | `{dataset}_pyramiddcn_s{seed}` | All four |
| Constant width | `{dataset}_width_ratio_1_matched_flops_s{seed}` | Criteo, KKBox, KuaiRec |
| Widening | `{dataset}_width_ratio_2_matched_flops_s{seed}` | All four |

All runs share the projected-input width, input connections, prediction head,
and training settings of the main tables. The default runs and the Criteo
constant-width runs are the ones reported in Tables 1 and 4.

## Files

| File | Contents |
| --- | --- |
| `runs.jsonl` | Configuration, test AUC and LogLoss, and FLOP count of each run |
| `summary.json` | Per-dataset means, sample SDs, FLOP ratios, and seed-paired differences |
| `per_seed.csv` | Seed-paired rows (default minus comparison) |

The comparisons use three seeds and no significance tests.
