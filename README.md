# PyramidDCN

Code for **PyramidDCN: Progressively Narrowing Cross Networks for CTR Prediction**.

This release contains the experiments reported in the paper: Tables 1–4 and
the width comparisons discussed in RQ4, all under the one-epoch,
projected-input protocol.

**Table 3 uses eight seeds (2024–2031) for every variant and its PyramidDCN
reference on all four datasets. Tables 1, 2, and 4 use three seeds (2024–2026).**

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
Table 3 is regenerated from 448 saved runs (14 conditions × 4 datasets ×
8 seeds), including its paired significance tests. Missing seeds cause an error.

## Train

Prepare a dataset using [data/README.md](data/README.md), then run:

```bash
python -m reproduce.run --table table1 --dataset criteo \
  --data-dir /path/to/prepared/criteo --device cuda \
  --output outputs/criteo.jsonl
```

Use `--table table2`, `table3`, or `table4` for the other
comparisons. Add `--seed 2024`, `--name RUN_NAME`, or `--limit 1` to select runs.
Supported devices include `cpu`, `cuda`, and `mps`.

To retrain Table 3, run the following for each of `criteo`, `avazu`, `kkbox`,
and `kuairec`, using the corresponding prepared data directory:

```bash
python -m reproduce.run --table table3 --dataset criteo \
  --data-dir /path/to/prepared/criteo --device cuda \
  --output outputs/criteo.jsonl
```

Without a `--seed` filter, this runs all 112 configurations for the selected
dataset: 14 conditions × 8 seeds. Use `--table all` on each dataset for a complete
rerun of Tables 1–4, then rebuild with
`python -m reproduce.build_tables --results-dir outputs`.

## Contents

| Path | Contents |
| --- | --- |
| `models/` | PyramidDCN and the nine baselines, one file per model |
| `train.py` | One-epoch training, evaluation, and FLOP counting |
| `configs/` | Dataset metadata and exact experiment configurations |
| `data/` | Dataset preparation scripts |
| `results/` | Saved metrics and reference tables |
| `reproduce/` | Run launcher and table generation |
| `tests/` | Synthetic forward, backward, and training checks |

## Models

The `model` field of a configuration is a class name in `models/`, and `args`
holds the keyword arguments of that class.

| Paper | File | Classes |
| --- | --- | --- |
| PyramidDCN | `models/pyramiddcn.py` | `PyramidDCN` |
| DCNv2 | `models/dcnv2.py` | `DCNv2` |
| ML-DCN | `models/mldcn.py` | `MLDCN`, `PyramidMLDCN` |
| MaskNet | `models/masknet.py` | `MaskNet`, `PyramidMaskNet` |
| QNN-α | `models/qnn.py` | `QNN`, `PyramidQNN` |
| ECN | `models/ecn.py` | `ECN`, `PyramidECN` |
| FCN | `models/fcn.py` | `FCN`, `PyramidFCN` |
| GDCN | `models/gdcn.py` | `GDCN` |
| FinalMLP | `models/finalmlp.py` | `FinalMLP`, `PyramidFinalMLP` |
| FiBiNet++ | `models/fibinetpp.py` | `FiBiNetPP`, `PyramidFiBiNetPP` |

`Pyramid<Model>` is the pyramidal conversion of that baseline in Table 2. DCNv2
and GDCN take per-layer widths, so their conversions use the same classes with
narrowing widths. The Table 3 variants and the Table 4 width schedules are
arguments of `PyramidDCN`. Run names are listed in
[the experiment protocol](docs/EXPERIMENTS.md#run-names).

See [the experiment protocol](docs/EXPERIMENTS.md) for hyperparameters and
comparison settings. The
[RQ4 width comparisons](results/reference/rq4_width_comparisons/README.md)
include their saved runs and a recomputation script.

All models share the projected-input protocol and the same training settings;
baselines are not tuned separately. FiBiNet++ uses groups of eight projected
input channels in place of semantic field embeddings. The Criteo encoder in
`data/` was reconstructed from the metadata of the cache used in the
experiments (see [data/README.md](data/README.md)). Raw data and checkpoints
are not included.

```bash
python -m unittest discover -s tests -v
```

Baseline sources and license notices are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
