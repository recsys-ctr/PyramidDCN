# Saved results

The four dataset JSONL files contain 610 distinct selected runs. Each record
contains its runnable configuration, test AUC, test LogLoss and linear FLOPs.
`reproduce.matrix_flops` adds direct bilinear contractions where applicable.

Rebuild CSV summaries, LaTeX table rows and paired/Holm tests:

```bash
python -m reproduce.build_tables
python -m reproduce.verify_tables
```

References in `reference/` correspond to this release. `manuscript_statistics.csv`
is the independent 99-comparison manuscript export used for verification.
`matrix_flops.csv` contains synthetic-forward checks at the actual widths.
RQ4's additional dataset comparisons remain in `reference/rq4_width_comparisons/`;
they are not substituted into Table 4. No raw data or checkpoints are included.
