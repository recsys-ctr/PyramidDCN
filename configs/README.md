# Configurations

Each `{table}/{dataset}.json` contains complete run configurations.
`all/{dataset}.json` is their deduplicated union. `model` selects a class from
`models/`; `args` contains constructor keywords. `input_proj` is the shared
projected-input width. Seeds and training settings are explicit per run.

`table_runs.csv` maps every table condition and seed to a configuration.
`table2_architectures.csv` summarizes the 24 selected Table 2 architectures.
`reporting_policy.json` fixes seed counts and statistical families.
`dataset_metadata.json` records field order, vocabulary sizes and split counts.
`table3_specification.json` maps displayed controls to same-width and
matched-FLOPs runs; a null matched arm means that cell was not evaluated.

Do not combine configurations or results from an older revision: several
baselines and their selected Table 2 conversions have changed.
