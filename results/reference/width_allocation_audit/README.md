# Historical width-allocation audit

This folder makes 33 existing normalized fixed-slice runs available for
recomputing comparisons omitted from the Criteo-only width table. No training
or inference was added. The canonical results files elsewhere in the
repository are unchanged.

From the repository root, run with Python 3.10 or newer:

```sh
python3 results/reference/width_allocation_audit/recompute.py
```

The script has no third-party dependency or network access. It resolves inputs
relative to itself, checks frozen run hashes and stored configurations, reconstructs matrix
profiles, verifies the reference's balanced-selector equivalence, and
recomputes the included JSON/CSV. `--write` regenerates those summaries.

## Included comparisons

The halving references are the current main-table runs named
`fair_{dataset}_v3n_stagger_s{seed}`. At their widths, the stored `stagger`
selector is exactly equivalent to `balanced`. Constant and increasing-width
controls use `balanced`, the same projected-input width, LN, input skip,
last-state head, one-epoch optimizer settings, and dataset cache. All three
individual seeds and all configurations are retained.

There are 12 references, nine completed near-budget constant controls
(Criteo, KKBox, KuaiRec), and 12 near-budget increasing-width controls
(all four datasets). The Avazu balanced constant control is present in the
historical plan but has no completed record in this audit. It is not imputed.

The Criteo constant `[344,344,344]` rows are exactly the current width-table
controls. Other-dataset constant controls and the increasing-width runs were
not previously included in the canonical result files. They are added here
with original source receipts rather than changing those files.

The existing data support **task-dependent allocation**: halving has better
observed metrics than reverse allocation on Criteo and Avazu; the KKBox and
KuaiRec comparisons favor more late-layer capacity. Steeper narrowing is not
uniformly beneficial. The summary reports each dataset separately.

## Files and limits

`runs.jsonl` contains the 33 original result lines. `provenance.json` supplies
source basenames, original line indices, source-file and record hashes,
stored arm configurations, and resolved defaults. Each copied line's hash is
verified directly. Original complete source files are not shipped in this
folder; their hashes are provenance receipts, not execution attestations.
`summary.json` and `per_seed.csv` are deterministic recomputations.
`SHA256SUMS` covers all source and data files in this folder.

These were adaptive historical experiments, with three seeds and one epoch.
Pairing is by seed and data-order convention, not identical initialization of
different matrix shapes. Matching stored configurations, cache identities,
profiles, and documented source semantics does **not** establish historical
executable, environment, or data-byte identity: those per-run fingerprints
were not saved. No new significance test, statistical equivalence, convergence,
causal mechanism, or universal accuracy claim is made. Matrix profiles include
the learned input projection and prediction head, and exclude embedding
lookups, normalization, and elementwise operations; they do not measure latency.
