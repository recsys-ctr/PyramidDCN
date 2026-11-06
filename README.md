# PyramidDCN

Code for **PyramidDCN: Progressively Narrowing Cross Networks for CTR Prediction**.

PyramidDCN uses a wide first cross layer and progressively narrows later layers.
Fixed, width-matched slices of the projected input provide the multiplicative
input and input skip connection without learned per-layer input projections.
The default cross-layer widths are `[W, W/2, W/4]`.

## Installation

Use Python 3.11+ and install PyTorch for your device, then:

```bash
python -m pip install -r requirements.txt
```

## Reproduce the paper tables

The saved results are included; no dataset download or retraining is needed:

```bash
python -m reproduce.build_tables
python -m reproduce.verify_tables
```

Tables are written to `outputs/tables/`. FLOPs count matrix multiplications,
including the input projection, prediction head, and bilinear contractions.

## Train

Prepare a dataset using [data/README.md](data/README.md), then run:

```bash
python -m reproduce.run --table table1 --dataset criteo \
  --data-dir /path/to/prepared/criteo --device cuda \
  --output outputs/criteo.jsonl
```

Use `--table table2`, `table3_core`, `table3_full`, or `table4` for the other
comparisons. Add `--seed 2024`, `--name RUN_NAME`, or `--limit 1` to select runs.
Supported devices include `cpu`, `cuda`, and `mps`.

## Contents

| Path | Contents |
| --- | --- |
| `experiments/` | Models, training, evaluation, and FLOP counting |
| `configs/` | Dataset metadata and exact experiment configurations |
| `data/` | Dataset preparation scripts |
| `results/` | Saved metrics, reference tables, and run provenance |
| `reproduce/` | Run launcher and table generation |
| `tests/` | Synthetic forward, backward, and training checks |

See [the experiment protocol](docs/EXPERIMENTS.md) for hyperparameters and
comparison settings, [per-dataset ablations](docs/TABLE3_BY_DATASET.md), and
[additional sensitivity experiments](docs/SENSITIVITY_PROTOCOL.md).

The main results use a shared projected-input protocol, without independent
baseline tuning. FiBiNet++ uses projected-coordinate groups in that protocol;
its semantic-field comparison is reported separately. The Criteo encoder was
reconstructed and checked on sampled cache correspondences; full-cache
equivalence has not been established. Raw data and checkpoints are not included.

```bash
python -m unittest discover -s tests -v
```

Baseline sources and license notices are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
