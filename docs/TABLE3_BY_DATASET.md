# Per-dataset ablations

The CSV files expand Table 3 into 56 dataset-condition rows: eleven core
conditions and three near-budget variants, each on four datasets.

```bash
python -m reproduce.export_table3_by_dataset --repo-dir . --out-dir /tmp/table3_export
```

The output directory must be outside the input repository. The exporter uses
the Python standard library and does not train models.

- `table3_per_dataset.csv`: AUC, LogLoss, sample SDs, run IDs, widths, and FLOPs.
- `table3_four_dataset_aggregates.csv`: the fourteen condition-level aggregates.
- `table3_export_validation.json`: source hashes and reference-table checks.

Each comparison uses the arm/reference seed intersection shared by all four
datasets, with three or eight seeds. Sample SD measures seed variation on the
fixed splits. Relative changes are `100 * (variant_mean / reference_mean - 1)`;
the table averages these four within-dataset changes rather than pooling AUCs.
Positive AUC changes and negative LogLoss changes favor the variant.

Near-budget variants use 97.9–99.8% of reference matrix FLOPs because widths are
discrete. Exact costs and matching reference cohorts are provided per row.
The exporter recomputes summaries with Decimal arithmetic and checks the core
and full reference table cells. These are descriptive results, not new fits.
