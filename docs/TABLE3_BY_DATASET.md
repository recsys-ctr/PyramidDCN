# Per-dataset Table 3 descriptive results

This package expands the existing, frozen Table 3 core comparisons into one row
per dataset. It adds no predictive training or inferential tests. It is an
auxiliary CSV export, not an additional manuscript table.

## Reproduce

Python 3.11 or later is sufficient; no third-party packages are required.

```sh
python -m reproduce.export_table3_by_dataset --repo-dir . --out-dir outputs/table3_by_dataset
```

`--repo-dir` must point to the repository containing `configs`, `results`, and
the frozen public reference tables. The exporter reads only the four named
historical JSONL files, two configuration files, and two reference TeX tables.
It does not import the training code or modify the input repository. No network
access is used. It fails if the frozen input set does not contain 798 records,
if selected model families or cohorts differ, or if the public table checks fail.

## Files

- `table3_per_dataset.csv`: 56 rows: 11 core conditions and 3 reported
  near-budget variants, each on four datasets.
- `table3_four_dataset_aggregates.csv`: the 14 condition-level descriptive
  aggregates used to check the existing Table 3 numbers.
- `table3_export_validation.json`: input hashes, cohort membership, numeric
  checks, comparisons with both frozen public reference tables, and cost scope.
- `export_table3_by_dataset.py`: the portable exporter and independent checker.

## Cohorts and interpretation

The exporter follows `reproduce/build_tables.py`: for each condition, it takes
the seed intersection shared by that arm and the reference over **all four
datasets**, restricted to 2024-2031. Each arm uses three or eight seeds. Every
comparison uses its own matching reference cohort. In particular, three-seed
arms are compared with three-seed references, not the eight-seed reference
summary shown in the reference arm's own row. Thus reference means and sample
standard deviations can differ between CSV rows by design.

Means and sample standard deviations describe seed variability on the fixed
dataset splits. Sample SD uses denominator `n - 1`; it is not a standard error
or a confidence interval. The fixed datasets are not treated as independent
seed replications. No missing results are imputed, and no seeds are selected by
outcome value.

`auc_absolute_change` is AUC minus its matching reference mean.
`auc_relative_change_percent` is `100 * (AUC / reference_AUC - 1)`.
The corresponding LogLoss changes use the same formulas; positive AUC changes
and negative LogLoss changes favor the variant. These relative percentages are
not percentage-point differences. The aggregate AUC change is the arithmetic
mean of the four **within-dataset relative changes**, as in the existing table;
it is not the relative change of a pooled cross-dataset mean. Widths, run names,
seeds, sample sizes, and matching reference identifiers are supplied for each
condition. Full numeric precision is retained in the CSV.

## Computation scope

All selected historical arms have model family `pml_mv`, for which the public
matrix-FLOP helper returns the logged `serving_linear_flops_2x` value unchanged.
The count covers matrix products in the common input projection, backbone, and
head; it excludes embedding lookups, normalization, and elementwise operations.
It is not latency or an energy estimate. Architecture and FLOP counts are checked
to be constant across the seeds within each condition. Historical dense-parameter
counts are deliberately neither used nor exported.

The three variants labeled as matched FLOPs in the existing table actually use
**97.9%-99.8%** of reference matrix FLOPs across their 12 dataset-condition
combinations because widths are discrete. They are labeled `near_budget` in
this export; exact per-dataset costs and ratios are included. This terminology
does not change the existing experimental configurations or results.

## Validation and disclosure boundary

The exporter independently recomputes means and sample SDs with 45-digit Decimal
arithmetic and independently derives cohorts by set intersection. It then reads
back the CSV, checks the four-dataset normalization, and verifies all reported
numeric cells against both frozen public core and full Table 3 files.

Only historical scalar experiment summaries and public configuration/table
metadata are processed. No dataset examples, checkpoints, new predictive runs,
external services, author identities, private review text, or review scores are
included. The run names are technical reproducibility identifiers. All recorded
paths are relative to the input repository. The source JSONL records contain
additional fields, but internal provenance, timing traces, and parameter-count
fields are not copied into this export. No manuscript claims are changed by
generating these descriptive artifacts.
