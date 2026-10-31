# Fixed sensitivity controls: protocol and reproduction

These controls supplement the original shared-input, one-epoch comparisons; they do not replace historical results or establish independently tuned state of the art. The initial twelve fits compare fixed selection with an existing learned-map recipe at near-matched matrix-FLOP cost. The secondary six fits compare native-field FiBiNet++ with the six new PyramidDCN selection fits under the same added training protocol.

## Freeze chronology and inspection policy

1. The twelve original configurations and initial analysis were frozen before the first new fit.
2. The six native-field configurations and secondary analysis were frozen after the initial twelve-fit queue had started, before any human inspection of new test metrics. This is a separate extension, not an eighteen-fit plan jointly frozen before the first fit.
3. Automatic full-validation checkpoint selection and one full-test evaluation per selected checkpoint are allowed. Do not open metric-bearing artifacts or analyze results until both queues, all eighteen fits, complete. Monitor only metric-free status/progress.
4. Both public analysis entry points enforce the mandatory eighteen-fit gate. The gate file must be present, enabled, and specify both exact queues and completed run rosters. A missing gate, partial queue, failure, missing run or changed roster is refused before metric files are opened. There is no partial-metric analysis option.

Original arm JSONs remain byte-identical, including historical arm identifiers. Public English planning documents and sanitized manifests were created later; their bytes are not claimed to predate training. Original private planning-document/manifest/config hashes remain as opaque provenance commitments. Non-scientific internal planning context is excluded.

## Fixed training and selection

Use the unchanged released `experiments/x4_runner.py` and existing complete Criteo/Avazu caches. Metadata, array shape and full-file hashes are verified before training. No sampling, row cap, new preprocessing or hidden diagnostic fits. The original Criteo preprocessing provenance limitation remains.

All new fits use Adam LR 0.0003, two complete training passes, batch 4096, embedding dimension 16, embedding LR multiplier 1, initialization SD 0.0001, embedding regularization 0, gradient clip 10, no scheduler and no erank/funnel/agop diagnostics. Execution is MPS float32, PyTorch intra-op threads 4/inter-op threads 1, with no CPU fallback. The launcher refuses a machine without MPS rather than changing the device silently.

Each epoch's full-validation AUC and LogLoss are recorded. A checkpoint replaces the previous best only when validation AUC is greater by more than 1e-6; otherwise retain the earlier epoch. Restore the chosen checkpoint and evaluate the complete test once. LogLoss and test metrics never select a checkpoint. The inherited runner may decrease LR after validation at the second endpoint, but no later training occurs; both actual training passes retain LR 0.0003.

Three fixed seeds are used per dataset/arm. The inherited seed*1000+epoch minibatch permutation is shared for the same dataset/seed/epoch, but differing model constructions do not imply bitwise-identical initialization. No automatic retry is implemented. Preserve numerical failures; any documented infrastructure retry must keep the same configuration and a separate attempt record. Do not hide incomplete fits or replace unfavorable outcomes.

## Reproduce training

The following commands assume these reproduction files are present in a fresh copy of the code release **before** either queue starts. Install the repository requirements and prepare the matching caches according to `data/README.md`. Do not modify code between queues.

From the repository root:

```sh
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

The second launcher can also wait for the first queue's completion using its required predecessor argument. It verifies the predecessor's twelve-run completion count, frozen plan and cache hashes. Use new output directories; a pre-existing queue status causes refusal rather than a silent repeat. Launchers redirect the harness log into the run directory and print metric-free progress; do not inspect the harness logs while the global gate is closed.

The shared trainer remains unchanged. The complete run loops, evaluation wrappers, checkpoint semantics, loss/optimizer inputs, full-precision evaluation returns, parameter arithmetic and descriptive-statistics computation match the original scripts. See `provenance/sensitivity/PORTABILITY.json` for exact original/public script hashes, byte-identical configuration hashes and AST checks. Adaptations affect public metadata/hash guards, mandatory global inspection checks, policy text and English report prose.

## Reproduce post-hoc statistics

**All eighteen fits completed without failures or retries.** The guarded raw bundle is released under `results/sensitivity`, with sibling `robustness_training` and `native_input_training` directories. `EXPORT_RECEIPT.json` records exact raw-file hashes and metadata adaptations; `COMPLETE.json` records the completed export. It must include the enabled parent `robustness_inspection_gate.json`, complete queue/per-run statuses, unchanged executed `config.json`, full-precision `evaluations_full_precision.json`, `model_profile.json`, `results_rounded.jsonl`, provenance and matching public `frozen_plan` snapshots. Preserve the original private-plan and source-record hashes in export provenance. Do not publish private planning documents or raw datasets.

For a rerun made directly with these public scripts, the output already uses public plan snapshots. For the original author execution, replace only the private planning-document snapshot with the matching sanitized public snapshot during export; retain original snapshot file hashes as original execution provenance. Keep raw evaluation/config/profile records unchanged where they contain only the documented scientific fields, and record any metadata-only sanitation separately. The analysis rejects private-plan snapshots whose bytes do not match the public hash guards.

Once the gate passes:

```sh
python reproduce/sensitivity/summarize_native_input_training.py \
  --original-run-dir outputs/sensitivity/robustness_training \
  --native-run-dir outputs/sensitivity/native_input_training \
  --out outputs/sensitivity_analysis
```

This validates and emits both comparisons. To regenerate only the initial learned-map comparison, with the **same mandatory eighteen-fit gate**:

```sh
python reproduce/sensitivity/summarize_robustness_training.py \
  --run-dir outputs/sensitivity/robustness_training \
  --out outputs/sensitivity_analysis/learned_map_control
```

The standard-library analysis reads full-precision per-epoch validation and selected-checkpoint test returns, checks the inherited >1e-6 checkpoint rule, complete row/step/evaluation counts, exact configurations, float32/MPS device records, rounded/full-precision agreement, parameter decomposition and matrix FLOPs. Native-field matrix FLOPs include custom bilinear products: twice Linear MACs plus 2*(F*D*D+F*F*D), with D=16.

Each comparison reports all six per-seed AUC and LogLoss differences, separate per-dataset means/sample SDs, arm summaries, validation trajectories and selected epochs. Positive AUC and negative LogLoss differences favor PyramidDCN selection. No pooled primary estimate, p-values, general superiority, equivalence/noninferiority or post-hoc equivalence margin. The two comparisons reuse the same six new PyramidDCN observations and are not independent replications.

## Interpretation limits

The initial control changes LR, epoch cap and checkpoint policy together; their effects are not causally separated. Selection and learned-map arms have different state widths, so the comparison is a near-budget recipe comparison rather than a pure adapter ablation. Existing test splits are reused, and two epochs do not establish convergence.

The secondary native-field comparison is descriptive under the added protocol. Removing input projection changes field grouping, bilinear operations, DNN input width and parameter/FLOP budgets. There are no new projected-FiBiNet++ fits under this protocol; old projected Table 1 runs differ in LR, epoch, device and seed, so a pure projection effect cannot be inferred. Exact Table 1 hand-port DNN overrides remain; this is not an independently tuned official reproduction or native-input SOTA claim.

Report direction reversals, mixed seed signs, AUC/LogLoss tradeoffs, failure counts and tiny differences without changing the plan. The released `results/sensitivity/analysis` reports every observed outcome. PyramidDCN improves over the near-budget learned-map recipe on both datasets; native-field FiBiNet++ has better AUC and LogLoss on Criteo, while PyramidDCN is better on Avazu. Criteo runs all select the second epoch and Avazu runs all select the first. These results establish neither convergence nor native-input dominance.

To regenerate the released summaries without training, use the same analysis command with `--original-run-dir results/sensitivity/robustness_training --native-run-dir results/sensitivity/native_input_training --out /path/to/new-analysis`. The `PORTABILITY.json` scope describes the earlier code-and-plan preparation snapshot; the later result export is documented separately by `EXPORT_RECEIPT.json`.
