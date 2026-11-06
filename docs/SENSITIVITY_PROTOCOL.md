# Sensitivity experiments

Eighteen additional fits on Criteo and Avazu compare PyramidDCN with a
near-budget learned-input-map variant and semantic-field FiBiNet++.
Each comparison has three seeds per dataset (2032–2034), sharing the six
PyramidDCN fits. These results are separate from the four main tables.

## Protocol

All fits use two complete epochs, Adam learning rate 0.0003, batch size 4096,
embedding dimension 16, embedding LR multiplier 1, initialization SD 0.0001,
no embedding regularization, and gradient clipping at 10. Checkpoints are selected
by full-validation AUC: replace the previous checkpoint only for an improvement
greater than `1e-6`; otherwise keep the earlier epoch. The selected checkpoint
is evaluated once on the full test split. Both training passes use LR 0.0003.

The initial twelve-fit configuration and analysis plan preceded training. The
six-fit semantic-field extension was fixed after the initial queue started,
before inspection of the new test results. This is not a jointly preregistered
18-fit study. Public protocol files were prepared later; original plan hashes
and unchanged arm configurations retain their provenance. The launchers check
source and cache hashes, preserve failures, and permit analysis only after both
queues finish. All eighteen released fits completed without failures or retries.

The original execution used MPS float32. Its launchers retain that device
requirement and refuse to silently fall back to another device. The main
experiment launcher supports other PyTorch devices.

## Reproduce training

Prepare the datasets as described in [data/README.md](../data/README.md), then:

```bash
mkdir -p outputs/sensitivity
cp configs/sensitivity/robustness_inspection_gate.json outputs/sensitivity/

python reproduce/sensitivity/launch_robustness_training.py \
  --code-dir . --plan-dir configs/sensitivity/robustness_plan \
  --criteo-data-dir /path/to/prepared/criteo \
  --avazu-data-dir /path/to/prepared/avazu \
  --out outputs/sensitivity/robustness_training

python reproduce/sensitivity/launch_native_input_training.py \
  --code-dir . --plan-dir configs/sensitivity/native_input_plan \
  --criteo-data-dir /path/to/prepared/criteo \
  --avazu-data-dir /path/to/prepared/avazu \
  --requires-completed-queue outputs/sensitivity/robustness_training \
  --out outputs/sensitivity/native_input_training
```

Use new output directories and keep model code unchanged between queues.
Exact configurations and checkpoint rules are also in the plan directories.

## Rebuild summaries from saved results

```bash
python reproduce/sensitivity/summarize_native_input_training.py \
  --original-run-dir results/sensitivity/robustness_training \
  --native-run-dir results/sensitivity/native_input_training \
  --out outputs/sensitivity_analysis
```

For newly trained results, replace `results/sensitivity` with
`outputs/sensitivity`. The analysis verifies configurations, checkpoint
selection, evaluation counts, full-precision metrics, and FLOPs. It reports all
per-seed differences and per-dataset means/sample SDs, without significance or
equivalence claims. Raw records and released summaries are in
`results/sensitivity/`; export hashes are in `EXPORT_RECEIPT.json` there.

## Interpretation

PyramidDCN outperforms the tested learned-map recipe on both datasets.
Semantic-field FiBiNet++ has better AUC and LogLoss on Criteo, while PyramidDCN
is better on Avazu. All Criteo runs select epoch two; all Avazu runs select epoch
one. The summaries report both datasets and all seed outcomes.

The learned-map comparison changes state widths to approximate the FLOP budget,
so it is not an isolated adapter ablation. The semantic-field comparison changes
field grouping, bilinear operations, DNN input width, and computation. It does
not isolate projection removal, and the baseline was not independently tuned.
The added protocol changes learning rate, training duration, and checkpoint
selection together. Reused test splits, two epochs, and shared PyramidDCN runs
do not establish convergence, independent replication, or native-input dominance.
