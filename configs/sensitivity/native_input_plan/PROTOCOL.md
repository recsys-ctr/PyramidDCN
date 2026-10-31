# Separately frozen native-field FiBiNet++ plan: secondary 6 fits

These six configurations and their analysis were frozen after the initial twelve-fit queue had begun, before any human inspection of new test metrics. They are a separate addition, not part of an eighteen-fit plan jointly frozen before the first fit. The initial twelve configurations, order, checkpoint rule and single test evaluations remain unchanged. All human metric inspection and analysis wait until BOTH queues, all eighteen fits, finish.

This public English document is a later translation/adaptation of private planning metadata. Original private document/manifest hashes remain in manifest.json; arm JSONs and the synthetic shape-audit record are byte-identical to the original frozen files. The public metadata bytes themselves are not claimed to predate training.

## Question and controls

Compare native-field FiBiNet++ with the six new PyramidDCN selection fits under the same added LR/epoch/device/seed protocol. This is a descriptive baseline comparison, not a pure projection-removal causal ablation. There are no new projected-FiBiNet++ fits under this protocol. Old Table 1 projected-FiBiNet++ differs in LR, epoch, device and seeds, so its change cannot identify the effect of removing projection. This is not an independently tuned official reproduction or native-input SOTA test.

Starting from the released Table 1 per-dataset FiBiNet++ configurations, remove input_proj and retain the exact DNN overrides: Criteo [512,512,512] and Avazu [312,312,312], rather than substituting constructor defaults. Feed native Criteo 39 fields x16 and Avazu 24 fields x16. Retain low_rank=50, SENet groups=2, reduction ratio=3, embedding batch normalization and all remaining released initialization/activation/dropout choices. Removing projection also changes field grouping, bilinear contraction, DNN input width and parameter/FLOP budget; this is not a matched-budget control.

Use seeds 2032, 2033 and 2034, in that order for Criteo then Avazu: six new fits, each with two full epochs. Reuse the six already fixed new PyramidDCN test evaluations for the comparison; do not rerun them. The two analyses share PyramidDCN observations and are not independent replications.

Use the same full caches, Adam LR 0.0003, embedding LR multiplier 1, batch 4096, initialization SD 0.0001, embedding regularization 0, gradient clip 10, train_frac 1, no scheduler and no erank/funnel/agop diagnostics. Use unchanged released run_arm on MPS float32, threads 4/interop 1 and no CPU fallback. Check full metadata/cache-file hashes against the initial queue before training. No sampling or preprocessing change. Training/validation/test row counts are those in the initial plan.

Retain the inherited validation AUC > best+1e-6 checkpoint update; otherwise keep the earlier epoch. Restore that checkpoint and evaluate the full test once. No test/LogLoss checkpoint choice, forced epoch-2 selection or altered minibatch ordering. Equal seed numbers do not imply bitwise-identical model initialization across architectures.

## Analysis

Primary: all six paired test AUC differences new PyramidDCN minus native FiBiNet++, plus separate per-dataset means/sample SDs of three differences and each arm's mean/sample SD. Secondary: paired LogLoss at the same checkpoint; negative favors PyramidDCN. Report both epoch validation metrics, selected epoch, actual rows/steps, runtime, full/non-embedding parameter counts and complete matrix FLOPs.

Do not pool the datasets into a primary estimate, calculate p-values, claim general superiority/equivalence/noninferiority, or call the eighteen fits eighteen independent comparison observations. Disclose reversals, mixed seed signs and AUC/LogLoss tradeoffs. Describe favorable direction in every seed only if all six measured AUC differences are positive. Two epochs do not establish convergence; existing benchmark test splits are reused.

Keep every failure and retry record. Numerical failures are outcomes; documented infrastructure retries may use only the same frozen configuration. The launcher has no automatic retry. Do not add projected-FiBiNet++ runs or retune LR/width/seeds after results. Freeze files remain unchanged; deviations are separate records.

## Synthetic audit

synthetic_shape_audit.json records a vocabulary-16 CPU inference-only dummy forward, not training/validation/test metrics. It verifies native field count, 16-dimensional embeddings, bilinear 16x16 matrix, F(F-1)/2 pairs, DNN input/hidden widths and matrix FLOPs. Full-vocabulary parameter totals are analytic values and are checked again on actual training graphs. Complete matrix FLOPs are twice Linear MACs plus 2*(F*D*D+F*F*D), with D=16.
