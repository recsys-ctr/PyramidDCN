# Experiment protocol

`configs/reporting_policy.json` fixes the seed counts and statistical policy.
`configs/table_runs.csv` maps each reported condition and seed to a runnable
configuration and saved result. Runs shared between tables are stored once.

| Group | Dataset | Runs | Scope |
| --- | --- | ---: | --- |
| `table1` | Each of four datasets | 27 each | Eight baselines and PyramidDCN |
| `table2` | Criteo | 75 | 24 baseline conditions and PyramidDCN |
| `table3` | Each of four datasets | 112 each | 14 conditions, including the reference |
| `table4` | Criteo | 21 | Seven width schedules |
| `all` | Four datasets | 610 total | Deduplicated selected runs |

The Original Criteo runs in Tables 1 and 2 are identical. In particular, both
now use the same dense DCNv2, serial MaskNet and parallel FCN_p configurations.

## Training

| Setting | Value |
| --- | --- |
| Epochs | 1; evaluate the epoch-end checkpoint |
| Batch size | 4096 |
| Optimizer / learning rate | Adam / 0.001 |
| Embedding initialization SD | 0.0001 |
| Embedding regularization | 0 |
| Gradient clipping norm | 10 |
| Learning-rate schedule | None |
| Seeds, Tables 1/2/4 | 2024, 2025, 2026 |
| Seeds, Table 3 | 2024–2031 |
| Embedding dimension | 16 per field |
| Default projected-input widths | Criteo 512; Avazu 312; KKBox 248; KuaiRec 472 |

Training examples are shuffled with a separate generator seeded by
`seed * 1000 + 1`, so the minibatch order matches across models for a seed.
Models learn their own input projection. The same data splits and optimizer
settings are used; the baselines are controlled backbone implementations,
not independently tuned reproductions of their original benchmark results.

Model-specific objectives are preserved: FCN_p uses probability-space Tri-BCE;
QNN uses its self-ensemble objective. Other selected models use binary
cross-entropy with logits. FCN_p has two parallel three-layer paths, not a
serial ECN–LCN stack. FiBiNet++ treats groups of eight projected channels as
fields. See [TABLE2_MODELS.md](TABLE2_MODELS.md) for the exact adaptations.

## Statistics and aggregation

AUC and LogLoss use arithmetic means and sample standard deviations across
seeds. The builder uses matching seed IDs for two-sided paired t-tests and
Holm correction at 0.05, implemented once in `reproduce/statistics.py`.
Incomplete or non-finite pairs are rejected. Families are fixed as follows:

| Table | Metric | Comparisons per family | Paired observations |
| --- | --- | ---: | ---: |
| 1 | AUC / LogLoss, separately | 32 each | 3 |
| 2 | AUC vs. Original | 16 | 3 |
| 3 | Relative AUC vs. PyramidDCN | 13 | 8 |
| 4 | AUC vs. PyramidDCN | 6 | 3 |

Table 3 reports the mean of four within-dataset FLOPs ratios and relative AUC
changes. For its test, each seed contributes the four-dataset average of
`100 * (variant_auc - reference_auc) / mean_reference_auc`. The denominator is
fixed to the observed eight-seed reference mean for each dataset. Thus the
test has eight observations, not 32 independent dataset/seed observations.
Blank matched-FLOPs cells are untested conditions, not missing seeds.

These tests measure variation across training seeds on fixed data splits.
Table 2 configurations were explored before selection; its tests are
exploratory and Holm covers the displayed comparisons, not the whole prior
architecture search. No samples are excluded to change significance.

## Matrix FLOPs

Counts are per example at two FLOPs per multiplication–addition pair. They
include the shared input projection, all invoked linear maps, the prediction
head and custom bilinear contractions. Embedding lookups, normalization,
reductions and elementwise products are excluded. The profiler checks both
linear and total matrix FLOPs against saved records using actual model shapes.

## Results

`results/{dataset}.jsonl` contains the selected per-seed metrics and exact
configuration fields. `results/reference/` contains the frozen table rows,
metric summaries, profiling results and statistics. Each saved result must
match its launcher configuration. Training outputs should go to `outputs/`,
not overwrite `results/`.

Only selected Table 1–4 runs and their dependencies are included. Unselected
architecture trials, additional Table 4 seeds and supplementary width studies
are excluded from this release.
