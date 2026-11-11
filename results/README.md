# Paper results

The four dataset JSONL files contain the 631 runs used in Tables 1–4.
Table 3 includes 448 runs: 14 conditions on four datasets with eight seeds
(2024–2031), including the reference. Tables 1, 2, and 4 use the three seeds
2024–2026. Runs shared between tables are stored once.
`reference/rq4_width_comparisons/` contains the 33 runs behind the additional
width comparisons discussed in RQ4 (some also occur in the main tables).

Each record holds the run configuration, test AUC and LogLoss, and the FLOP
count of its linear layers. All runs use one training epoch. Raw examples,
learned weights, and checkpoints are not included.

`reference/` contains the table rows, FLOP profiles, and statistical summaries:

| File | Contents |
| --- | --- |
| `tables/table{1,2,3,4}_rows.tex` | Table rows as printed in the paper |
| `main_table1_comparisons.csv`, `holm36_summary.json` | Welch tests with Holm correction for Table 1 |
| `table2_welch.csv` | Welch tests for Table 2 |
| `table3_significance.csv`, `table3_seed_averages.csv`, `table3_significance_summary.json` | Paired tests for Table 3 and their seed-wise inputs |
| `matrix_flops.csv`, `release_matrix_profiles.csv`, `table2_matrix_profiles.csv` | Matrix FLOP profiles |

Run `python -m reproduce.verify_tables` to reproduce Tables 1–4.
