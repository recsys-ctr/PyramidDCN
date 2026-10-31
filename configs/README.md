# Frozen experiment configurations

Each JSON in a group directory is a list of original run configurations for one dataset. `all/` contains their deduplicated union. `experiment_index.json` declares exact group membership, `dataset_metadata.json` gives public field/count metadata for shape checks and synthetic profiling, and `table3_specification.json` declares the full author-reference layer table.

The configuration records preserve the trained model, dimensions, learned-map variants, seed, initialization, optimizer settings, and diagnostic flags. They do not contain test results or learned weights. The matching scalar records and source hashes are in `results/`.

The main Table 1 DCNv2 representative is rank `floor(W/4)`. Table 2 uses a separate rank-W DCNv2 starting point, as stated in its caption. Do not replace these distinct settings with one another.
