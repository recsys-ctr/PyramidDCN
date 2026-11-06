# Experiment protocol

The release contains 798 saved runs across Criteo, Avazu, KKBox, and KuaiRec.
Configurations are in `configs/`; scalar results are in `results/{dataset}.jsonl`.
Reference runs shared between tables are stored once. Original run identifiers
and source hashes are retained in `results/provenance.json`.

## Comparisons

| Group | Dataset | Runs per dataset | Comparison |
| --- | --- | ---: | --- |
| `table1` | All four | 30 | Nine baselines and PyramidDCN |
| `table2` | Criteo | 84 | Original backbones and pyramidal conversions |
| `table3_core` | All four | 72 | Selected layer and input-connection ablations |
| `table3_full` | All four | 150 | Complete 31-alternative layer exploration |
| `table4` | Criteo | 21 | Seven width schedules |
| `text_no_ln` | All four | 6 | With and without layer normalization |
| `all` | All four | Varies | Union of configurations, without duplicates |

Table 1 uses factorized DCNv2 with rank `floor(W/4)`; Table 2 starts from rank
`W` and sets pyramidal ranks according to layer widths. These settings have
different run identifiers. Table 2 widths were chosen under the original
linear-layer FLOP cap and then reprofiled with complete matrix-product counts;
the released configurations reproduce those choices.

## Training and inputs

All main-table models use projected field embeddings and the same training
protocol. Baselines are adaptations to this harness, not independently tuned
reproductions. FiBiNet++ groups projected coordinates into size-eight vectors;
these groups are not semantic fields. See the separate sensitivity experiments
for its semantic-field input comparison.

| Hyperparameter | Main-table setting |
| --- | --- |
| Training epochs | 1 |
| Batch size | 4096 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Embedding initialization SD | 0.0001 |
| Embedding regularization | 0 |
| Gradient clipping norm | 10 |
| Learning-rate schedule | None |
| Seeds | 2024, 2025, 2026 |

Model-specific widths, embedding dimensions, and loss settings are explicit in
each JSON configuration. Five layer alternatives and their reference have eight
seeds, 2024–2031. Each ablation uses the seed intersection shared by its variant
and reference across all four datasets. Dataset splits and preprocessing are
described in [data/README.md](../data/README.md).

## Metrics and FLOPs

AUC and LogLoss summaries use means and sample standard deviations. Table 3
averages the four within-dataset relative AUC changes,
`100 * (variant_mean / reference_mean - 1)`, and FLOPs ratios. Its three-seed
and eight-seed comparisons therefore use different reference means. Layer and
width-schedule comparisons are descriptive. Table 2 stars denote exploratory,
unadjusted two-sided Welch tests (`p < .05`); additional Welch/Holm calculations
are exported by the table builder. Saved main-table metrics have six decimal
places; regenerating tables does not recover greater precision.

FLOPs count matrix multiplications at two operations per multiplication–addition
pair, including the input projection, prediction head, and direct bilinear
contractions. Embedding lookups, normalization, reductions, and elementwise
operations are excluded. These counts do not measure latency.

```bash
python -m reproduce.profile_flops --scope all --out outputs/matrix_flops.csv
```

The profiler uses synthetic categorical IDs while retaining all model shapes.
Table 4 reports width schedules, FLOPs, and AUC; effective-rank diagnostics are
not part of the current table. Diagnostic fields in the original records remain
available for traceability.

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

The [per-dataset ablation export](TABLE3_BY_DATASET.md) provides the matching
cohorts and exact FLOPs ratios. The [18 sensitivity fits](SENSITIVITY_PROTOCOL.md)
use a separate two-epoch protocol and are not pooled into the main tables.
