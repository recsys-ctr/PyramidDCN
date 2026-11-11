# Experiment configurations

Each JSON in a group directory (`table1`, `table2`, `table3`, `table4`) is a list of run configurations for one dataset. `all/` contains their deduplicated union. `experiment_index.json` declares exact group membership, `dataset_metadata.json` gives public field/count metadata for shape checks and synthetic profiling, and `table3_specification.json` declares the rows of Table 3.

The configuration records give the trained model, dimensions, input-connection variants, seed, initialization, and optimizer settings (one epoch, batch 4096, gradient clipping 10, no regularization). `model` is a class name in `models/`, `args` holds the keyword arguments of that class, and `input_proj` is the width of the shared projected input. Run names have the form `{dataset}_{arm}_s{seed}`; the arms are listed in [the experiment protocol](../docs/EXPERIMENTS.md#run-names). The records do not contain test results or learned weights. The matching result records are in `results/`.

`table3/` contains 112 configurations per dataset: 14 conditions with eight
seeds (2024–2031). `table1/`, `table2/`, and `table4/` use three seeds (2024–2026).
The group-specific seed lists are declared in `experiment_index.json` under
`seeds_by_group`. The run launcher executes all seeds unless explicitly filtered.

The Table 1 DCNv2 baseline (`dcnv2`) uses rank `floor(W/4)`. Table 2 starts from a separate rank-W DCNv2 (`dcnv2_full_rank`), as stated in the paper (RQ2). Do not replace these distinct settings with one another.
