# PyramidDCN

Code and selected experiments for **PyramidDCN: Progressively Narrowing Cross
Networks for CTR Prediction**. This revision reproduces the Table 1–4 settings
selected on 2026-11-15.

| Table | Experiments | Seeds |
| --- | --- | --- |
| 1 | Eight baselines and PyramidDCN on four datasets | 3 |
| 2 | Eight backbones: Original, same initial width, near-matched FLOPs | 3 |
| 3 | Input-connection and cross-layer controls on four datasets | 8 |
| 4 | Seven Criteo width schedules | 3 |

All runs train for one epoch. There are **610 distinct runs** after sharing
identical runs across tables. ECN is used inside FCN, not as a separate baseline.

## Install

Use Python 3.11+ and install PyTorch for your device, then:

```bash
python -m pip install -r requirements.txt
```

The tested package versions are in `requirements-tested.txt`.

## Rebuild the reported tables

Saved per-seed metrics are included. These commands require no data download
or retraining:

```bash
python -m reproduce.build_tables
python -m reproduce.verify_tables
```

`outputs/tables/` contains the table rows, metric summaries and all 99 statistical
comparisons. Every table uses two-sided paired t-tests with Holm correction
within its table and metric. Missing seeds or mismatched configurations fail
verification.

## Retrain

Prepare the public datasets following [data/README.md](data/README.md). Inspect
or run the frozen configurations:

```bash
python -m data.verify_cache --dataset criteo --data-dir /path/to/prepared/criteo
python -m reproduce.run --table table2 --dataset criteo --list
python -m reproduce.run --table table2 --dataset criteo \
  --data-dir /path/to/prepared/criteo --device cuda \
  --output outputs/criteo.jsonl
```

Use `table1`, `table3`, or `table4` for a single table. For a complete rerun,
run `--table all` once for each of `criteo`, `avazu`, `kkbox`, and `kuairec`, using
its own prepared dataset directory and `outputs/{dataset}.jsonl`. Then:

```bash
python -m reproduce.build_tables --results-dir outputs
```

`--seed 2024`, `--name RUN_NAME`, and `--limit 1` select subsets. Completed runs
with identical configurations are skipped. Training supports `cpu`, `cuda`,
and `mps`; numerical results may vary across devices and software versions.
Saved metrics and rebuilt tables are a separate, deterministic check.

## Configuration map

- [`configs/table_runs.csv`](configs/table_runs.csv): every table cell and seed
  mapped to its run name and configuration file.
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md): training, aggregation and tests.
- [`docs/TABLE2_MODELS.md`](docs/TABLE2_MODELS.md): selected architectures,
  input connections and model-specific conversion exceptions.
- [`configs/table2_architectures.csv`](configs/table2_architectures.csv): all
  24 Table 2 architecture settings and matrix FLOPs.
- `models/`: runnable implementations; `model` and `args` in each JSON select
  the class and its constructor arguments.

The selected DCNv2 is dense with three cross layers and a parallel three-layer
MLP. FCN is FCN_p (three ECN layers in parallel with three LCN layers). MaskNet
uses three serial MaskBlocks and a scalar head. QNN-α, FinalMLP and FiBiNet++
retain their selected earlier configurations; their conversion exceptions are
listed explicitly in the architecture document.

## Check implementations and FLOPs

```bash
python -m unittest discover -s tests -v
python tests/test_smoke.py --full-catalog
python -m reproduce.profile_flops --scope all --out outputs/matrix_flops.csv
```

The catalog check uses synthetic inputs with the actual model widths. Matrix
FLOPs include input projection, prediction head and bilinear contractions.
They exclude embedding lookups, normalization and elementwise operations.

The Criteo encoder is reconstructed from the experimental cache metadata;
see its verification limits in [data/README.md](data/README.md). Raw data and
checkpoints are not distributed. Baseline sources and licenses are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
