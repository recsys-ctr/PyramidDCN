# Experiment protocol

The release contains 631 table runs across Criteo, Avazu, KKBox, and KuaiRec.
Configurations are in `configs/`; results are in `results/{dataset}.jsonl`.
Runs shared between tables are stored once.

## Comparisons

| Group | Dataset | Runs per dataset | Comparison |
| --- | --- | ---: | --- |
| `table1` | All four | 30 | Nine baselines and PyramidDCN |
| `table2` | Criteo | 84 | Original backbones and pyramidal conversions |
| `table3` | All four | 112 | Fourteen ablation/reference conditions, each with eight seeds |
| `table4` | Criteo | 21 | Seven width schedules |
| `all` | All four | Varies | Deduplicated configurations for Tables 1–4 |

Table 1 uses factorized DCNv2 with rank `floor(W/4)`; Table 2 starts from rank
`W` and sets pyramidal ranks according to layer widths. These settings have
different run names (`dcnv2` and `dcnv2_full_rank`).

## Run names

Run names have the form `{dataset}_{arm}_s{seed}`. Each configuration names its
model class in `model` (see `models/`) and that class's arguments in `args`.

| Table | Arms |
| --- | --- |
| 1 | `pyramiddcn`, `dcnv2`, `mldcn`, `masknet`, `qnn`, `ecn`, `fcn`, `gdcn`, `finalmlp`, `fibinetpp` |
| 2 | Original: the Table 1 baselines, with `dcnv2_full_rank` for DCNv2. Converted: `{model}_pyramid` (same initial width) and `{model}_pyramid_matched_flops` (near-matched FLOPs) |
| 3 | `pyramiddcn`, `input_pooling`, `frozen_projection`, `learned_projection`, `previous_layer_channels`, `addition_only`, `input_self_interaction`, `no_input_skip`, `projected_input_skip`, `shifted_input_skip`, `channel_overlap`; the suffix `_matched_flops` marks the matched-FLOPs column |
| 4 | `pyramiddcn` (ratio 1/2); `width_ratio_1`, `width_ratio_1_4`, `width_ratio_1_8` (same first width); the same three with the suffix `_matched_flops` (near-matched FLOPs) |

## Training and inputs

All models use projected field embeddings and the same training protocol.
Baselines are adapted to this harness and are not tuned separately.
FiBiNet++ uses groups of eight projected input channels in place of semantic
field embeddings.

| Hyperparameter | Setting |
| --- | --- |
| Training epochs | 1 |
| Batch size | 4096 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Embedding initialization SD | 0.0001 |
| Embedding regularization | 0 |
| Gradient clipping norm | 10 |
| Learning-rate schedule | None |
| Seeds: Tables 1, 2, 4 | 2024, 2025, 2026 |
| Seeds: Table 3 | 2024–2031 |

Model-specific widths, embedding dimensions, and loss settings are explicit in
each JSON configuration. Every Table 3 condition and its reference use all eight
seeds on every dataset (448 runs total). Dataset splits and preprocessing are
described in [data/README.md](../data/README.md).

## Metrics and FLOPs

AUC and LogLoss summaries use means and sample standard deviations. Table 1
comparisons use two-sided Welch t-tests with Holm correction across baselines
and datasets, separately for AUC and LogLoss. Table 2 stars denote unadjusted
two-sided Welch tests (`p < .05`).

Table 3 averages the four within-dataset relative AUC changes,
`100 * (variant_mean / reference_mean - 1)`, and FLOPs ratios. All comparisons
in Table 3 use the same eight-seed reference mean within each dataset.
For its significance tests, seed-paired AUC differences are divided by each
dataset's mean reference AUC and averaged across the four datasets. Two-sided
paired t-tests use these eight seed blocks, with Holm correction across all
13 non-reference comparisons; stars denote adjusted `p < .05`. The builder exports
`table3_significance.csv` and `table3_seed_averages.csv`. Saved metrics have
six decimal places.

FLOPs count matrix multiplications at two operations per multiplication–addition
pair, including the input projection, prediction head, and direct bilinear
contractions. Embedding lookups, normalization, reductions, and elementwise
operations are excluded.

```bash
python -m reproduce.profile_flops --scope all --out outputs/matrix_flops.csv
```

The profiler uses synthetic categorical IDs while retaining all model shapes.

## Commands

```bash
python -m reproduce.run --table table4 --dataset criteo \
  --data-dir /path/to/prepared/criteo --device cuda
python -m reproduce.build_tables --out-dir outputs/tables
python -m reproduce.verify_tables
```

`--output PATH` selects the result JSONL. Existing run names are skipped;
`--seed`, `--name`, and `--limit` filter pending runs. To build tables from a
complete set of reruns, supply `--results-dir outputs` to `build_tables`.
Every required configuration must be present; missing runs are not substituted.

The [RQ4 width comparisons](../results/reference/rq4_width_comparisons/README.md)
contain their runs and a recomputation script.
