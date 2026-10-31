# Released run records

`criteo.jsonl`, `avazu.jsonl`, `kkbox.jsonl`, and `kuairec.jsonl` contain the 798 unique saved records needed for the main paper, complete layer exploration, reported no-LN comparison, and constant-width effective-rank rerun. These are scalar metrics, model settings, histories, and small aggregate diagnostics. They contain no data rows, predictions, embedding weights, or checkpoints.

`provenance.json` records original JSONL file hashes and per-run relative source filenames/lines. Every released configuration was checked against its saved record and at least one original arm JSON definition.

`reference/` contains the manuscript's matrix-FLOP/statistical audit outputs and table-row snapshots. Paths in copied audit artifacts have been converted to relative provenance labels. Run `python -m reproduce.verify_tables` from the repository root to regenerate and compare the published table data.
