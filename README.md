# PyramidDCN

Research code for **PyramidDCN: Progressively Narrowing Cross Networks for CTR Prediction**.

PyramidDCN computes cross interactions in progressively narrower hidden states. Each layer reuses fixed coordinates of a shared learned input representation as its multiplicative input and input skip connection. The default three-layer schedule is `[W, W/2, W/4]`.

## What is included

- The experiment harness and model implementations used for the reported results, with machine-specific paths removed.
- Fixed configurations and saved scalar results for all four benchmarks, all four paper tables, the full exploratory layer-comparison table, and the additional normalization/effective-rank records used in the text.
- Dataset preprocessing scripts, training/evaluation, matrix-FLOP profiling, statistical recomputation, and table generation.
- Provenance linking each recorded run to its original configuration/result file and source hash.

The release freezes **798 distinct recorded runs**. It excludes raw datasets, embedding caches, trained checkpoints, private research-agent tooling, and manuscript drafts.

## Install

Use Python 3.11 or newer and install PyTorch for your hardware, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The CPU path works without an accelerator. Use `--device cuda` or `--device mps` when supported. See `docs/CORE_VALIDATION.json` for the environments used to verify the portable model code and `docs/EXPERIMENTS.md` for the experiment protocol.

## Reproduce tables from saved results

This recomputes the reported statistics and table rows without retraining:

```bash
python -m reproduce.build_tables
```

See `python -m reproduce.build_tables --help` for output options. Matrix-product FLOPs include the shared input projection, head, and custom bilinear contractions. They exclude embedding lookups, normalization, reductions, and elementwise operations and are not latency measurements.

## Train the recorded configurations

Prepare a benchmark following [data/README.md](data/README.md). A prepared dataset has `meta.json` and `{train,valid,test}_{cats,y}.npy` files. Then, for example:

```bash
python -m reproduce.run --table table1 --dataset criteo \
  --data-dir /path/to/prepared/criteo --device cuda \
  --output outputs/table1_criteo.jsonl
```

For one recorded arm, add `--name ORIGINAL_RUN_NAME`; `--seed 2024` selects a seed, and `--limit 1` limits the run count. Table groups are `table1`, `table2`, `table3_core`, `table3_full`, `table4`, `text_no_ln`, `effective_rank_rerun`, and `all`. Table 2, Table 4, and the effective-rank reruns are Criteo-only. The main-table configurations use one training epoch. Changing the training protocol creates a new experiment rather than reproducing the saved result.

The experiment harness is also available directly:

```bash
python -m experiments.x4_runner --data-dir /path/to/prepared/criteo \
  --arms configs/table1/criteo.json --device cpu --eval_test \
  --out outputs/direct_criteo.jsonl
```

## Interpretation and validation

These are controlled comparisons under a shared input design and training protocol. In particular, FiBiNet++ receives groups of projected coordinates rather than its native semantic fields. Interpret the results under that setup.

The portable core was checked against the original harness on seeded model initialization, logits, losses, gradients, and a tiny end-to-end training run. These checks verify the release implementation; they are not newly trained benchmark results. The original Criteo preprocessing source was unavailable: its replacement was reconstructed from the existing cache and checked on sample correspondence. The validation scope is documented in [data/README.md](data/README.md).

Run the data-free smoke test with:

```bash
python -m unittest discover -s tests -v
```

## Layout

| Directory | Contents |
| --- | --- |
| `experiments/` | Model, loss, training, evaluation, and diagnostic code |
| `configs/` | Complete recorded arm definitions and dataset metadata |
| `data/` | Preprocessing scripts and dataset instructions |
| `results/` | Saved scalar metrics and provenance; no examples or model weights |
| `reproduce/` | Run launcher, table/statistics generation, FLOP profiling |
| `docs/` | Protocol, validation, and implementation details |
| `tests/` | Synthetic smoke tests |

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for baseline sources and applicable notices.
