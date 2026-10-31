# Experiment reproduction

This release freezes the runs used by the manuscript's four main tables, its reported no-LN comparison, and the complete author reference table of 31 alternative layer forms. There are **798 unique saved runs** across the four datasets. Reused reference runs appear in several experiment groups but occur only once in each result JSONL and in `configs/all/`.

The release contains model definitions, training and evaluation, frozen run configurations, preprocessing, effective-rank and other recorded diagnostics, scalar results, matrix-operation accounting, and table/statistics regeneration. It contains no benchmark rows, embedding tables, trained checkpoints, or private agent workspaces.

## What each group reproduces

| `--table` group | Dataset | Runs per dataset | Purpose |
| --- | --- | ---: | --- |
| `table1` | all four | 30 | Nine baseline families plus PyramidDCN, three seeds each |
| `table2` | Criteo | 84 | Nine original backbones, two pyramidal conversions each, and PyramidDCN |
| `table3_core` | all four | 72 | Reference, ten selected alternatives, their reported matched-cost comparisons, and the corresponding seeds |
| `table3_full` | all four | 150 | Reference, all 31 alternative layer forms, and reported matched-cost variants |
| `table4` | Criteo | 21 | Seven schedules under the two width/budget comparison settings |
| `text_no_ln` | all four | 6 | Three no-LN runs and their three reference runs |
| `effective_rank_rerun` | Criteo | 3 | Constant-width effective-rank diagnostics from the original reruns |
| `all` | all four | varies | Union of all groups without duplicate run names |

The authoritative membership lists are in `configs/experiment_index.json`. Every configuration preserves its original run name and seed. `results/provenance.json` records the source JSONL filename, line, file hash, and matching original JSON configuration filenames. The file paths in provenance are relative to the original experiment directory; local machine paths have been removed.

### DCNv2 settings differ between Tables 1 and 2

Table 1 keeps the factorized DCNv2 representative at rank `floor(W/4)`, with original names `fair_{dataset}_v3f_r4_s{seed}`. Table 2's original DCNv2 is rank `W`, named `v3_criteo_dcnv2_lr_s{seed}`; its pyramidal ranks follow the layer widths. These are distinct configurations and both are included. The frozen settings should be used instead of inferring a Table 2 configuration from Table 1.

### Seeds and result units

The main comparisons and schedule table use seeds 2024–2026. Five layer alternatives and their reference have eight available seeds, 2024–2031, as marked by daggers in the complete layer table. Only the shared seeds present for every dataset and the corresponding reference enter an aggregate comparison.

Table 3 reports the arithmetic mean across four fixed datasets of the relative change from each dataset's reference mean AUC. It is a relative percentage, not an absolute AUC difference. The eight-seed reference used for a daggered comparison is distinct from the three-seed reference mean in Table 1. FLOPs ratios are also averaged over the four datasets. These layer and schedule results are descriptive in the current manuscript.

## Rebuild the reported tables without training

From the repository root, after installing the requirements:

```bash
python -m reproduce.build_tables --out-dir outputs/tables
python -m reproduce.verify_tables
python -m reproduce.profile_flops --scope main --out outputs/matrix_flops.csv
```

`build_tables` reads the released scalar records and writes four main table row files, full/core layer CSVs and TeX, schedule results, and exact Welch/Holm calculations. `verify_tables` compares the regenerated data rows with the frozen manuscript rows, including the full 31-alternative layer table. It also checks the saved Holm summary. It does not train or download data.

The main table's two-sided Welch tests use data-dependent degrees of freedom and a separate Holm family of 36 comparisons for each metric. Table 2's stars denote unadjusted exploratory two-sided Welch `p<.05`. The historical fixed `|t|>=2.78` shortcut is not used by the release table builder.

The original run records store test metrics rounded to six decimal places. Regeneration uses those same recorded values; it cannot recover predictions or greater numerical precision from a saved checkpoint because checkpoints are not distributed.

## Matrix-operation accounting

Reported FLOPs count matrix multiplications at two FLOPs per MAC, including the shared input projection, head, and custom bilinear contractions. Embedding lookups, normalization, reductions, and elementwise operations are excluded. They are not latency measurements.

`experiments/flops.py` reproduces the original `nn.Linear` hook count. `reproduce/matrix_flops.py` adds the direct bilinear products in FiBiNet++ and FinalMLP, including their pyramidal counterparts. `reproduce/profile_flops.py` independently checks these totals with PyTorch's dispatcher FLOP counter on CPU. It uses a 16-row synthetic vocabulary and zero categorical IDs while retaining the field count, projection width, matrix dimensions, and model settings. Reducing the synthetic vocabulary does not change a counted matrix product.

```bash
python -m reproduce.profile_flops --scope all --out outputs/all_matrix_flops.csv
```

`--scope all` checks every distinct exported model configuration, including the full layer exploration. Reference profiling outputs from the manuscript audit are preserved under `results/reference/`.

## Train and evaluate a frozen experiment group

Prepare one of the four datasets using the instructions in the main README. A prepared directory must contain `meta.json`, `train_cats.npy`, `train_y.npy`, and the corresponding validation/test arrays. The training code receives this directory explicitly; it does not infer a local user path.

```bash
python -m reproduce.run --table table1 --dataset criteo --data-dir data/criteo --device cuda
python -m reproduce.run --table table2 --dataset criteo --data-dir data/criteo --device cuda
python -m reproduce.run --table table3_core --dataset avazu --data-dir data/avazu --device cuda
python -m reproduce.run --table table3_full --dataset kkbox --data-dir data/kkbox --device cuda
python -m reproduce.run --table table4 --dataset criteo --data-dir data/criteo --device cuda
```

Use `--device cpu`, `mps`, or another PyTorch device supported by the machine. Device changes may affect numerical results. The frozen protocol is one epoch with batch size 4096, Adam learning rate 0.001, embedding initialization standard deviation 0.0001, no embedding regularization, gradient clipping at 10, and no learning-rate schedule. All further model-specific settings are explicit in the JSON configurations.

`--table all` runs the union for the selected dataset. Shared run names already present in the output file are skipped. By default records append to `outputs/{dataset}.jsonl`; `--output PATH` selects another file. `--seed 2024`, `--name ORIGINAL_RUN_NAME`, and `--limit 1` filter the pending runs. The limit selects a complete experiment run, not a reduced-data smoke test.

A lower-level entry point is also available:

```bash
python -m experiments.x4_runner --arms configs/table4/criteo.json --data-dir data/criteo --device cpu --eval_test --out outputs/criteo.jsonl
```

To rebuild tables from newly completed experiments instead of released records, place the four dataset JSONLs in a directory and pass it with `--results-dir`:

```bash
python -m reproduce.build_tables --results-dir outputs --out-dir outputs/retrained_tables
```

A complete table regeneration expects every required run, including the full layer exploration and effective-rank rerun. There is no automatic substitution of an unmatched seed or another model variant.

## Configuration provenance

The exported configurations were cross-checked against both the saved run record and an original JSON arm definition. In the training log, diagnostic dictionaries replace the `erank`, `funnel`, and `agop` boolean flags. The exported training configuration restores the original booleans rather than passing a result dictionary back into training.

The frozen configurations replace the need to rerun the historical sequence of exploratory generators or width searches. In particular, Table 2's original-budget widths were selected under the original linear-layer FLOP cap, then reprofiled using complete matrix-product counts. They should not be reselected under a new cost metric when reproducing the reported experiments. The original definition filenames and source-record hashes remain available for traceability.

The current core Table 3 contains eleven rows (reference plus ten alternatives), with compact formula aliases defined in the manuscript. Table 4 includes sample standard deviations in its AUC column and three per-layer effective-rank percentages; the redundant mean-rank column is omitted. The complete 31-alternative table remains available.

## Supplementary records, separate from the four main tables

The 798 historical records remain unchanged. The additional eighteen fits use two full epochs at learning rate 0.0003 on the existing Criteo/Avazu caches, with three new seeds. The six new PyramidDCN runs are shared by the learned-map and semantic-field FiBiNet++ comparisons; they are not counted twice. The initial twelve-fit plan was frozen before its first fit. The native-field six-fit plan was frozen after those fits started and before human inspection of any new test result. Human inspection and analysis require both queues to finish successfully.

See [SENSITIVITY_PROTOCOL.md](SENSITIVITY_PROTOCOL.md) for exact configurations, checkpoint-selection tolerance, input-path differences, guards, limitations and reproduction commands. The study does not establish convergence, pure projection-removal causality, an independently tuned original implementation, or an untouched confirmatory holdout. It does not alter Tables 1–4 or their historical statistical comparisons. Full-precision evaluations are provided for the supplementary fits; historical metrics retain their recorded precision.

[Table 3 by dataset](TABLE3_BY_DATASET.md) contains 56 descriptive dataset-condition rows from the historical records, not 56 new fits. Sample SDs describe seed variability, and each variant uses a matching reference cohort. The near-budget variants use 97.9–99.8% of reference matrix FLOPs.

The historical funnel diagnostics in `experiments/funnel_check.py` analyze the canonical identity-residual, low-rank DCNv2 cross stream while holding the input and parallel branch fixed. Historical theorem-number comments refer to that diagnostic's original context. They are not a lossless-compression or declining-activation-rank guarantee for the normalized PyramidDCN network.
