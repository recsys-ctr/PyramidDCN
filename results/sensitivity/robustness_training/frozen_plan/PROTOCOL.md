# Frozen alternative-protocol sensitivity plan: initial 12 fits

The original configuration lists and analysis were frozen before the first new fit. This public English document is a later translation/adaptation of private planning metadata. It is not a claim that these public document bytes were registered before training. The original private document/manifest SHA-256 commitments are retained in manifest.json; configuration JSONs are unchanged byte for byte.

Question: does the observed ordering of fixed input selection and a learned input map under a near-matched matrix-FLOP budget persist under one fixed alternative training protocol? This does not separately identify learning-rate versus epoch/checkpoint effects.

## Fixed arms

| Dataset | Selection | Learned input map | Shared input width |
| --- | --- | --- | ---: |
| Criteo | stagger, [512,256,128] | learned, [264,132,66] | 512 |
| Avazu | stagger, [312,156,78] | learned, [160,80,40] | 312 |

All use pml_mv, one view, no mask, t=1, affine LayerNorm, input reread and the same input addend, and a linear head on the final state. The stagger selectors coincide with least-used balanced selectors for these halving schedules. Learned-map widths are the existing fixed budget-control widths; there is no new width search. State widths differ, and the learned input map changes both multiplicative and additive inputs. This is a near-matched-cost recipe comparison, not a pure input-map causal ablation or function-class equivalence test.

Seeds 2032, 2033, and 2034 are used for both arms on both datasets: 12 fits, each with two complete training passes. The order in each dataset is seed then selection/learned_budget; Criteo precedes Avazu. Do not add seeds, replace unfavorable runs, or change arms in response to metrics.

## Data and training

Use the existing complete cached encoding, vocabulary, labels and train/validation/test splits. No sampling, row limit or new preprocessing is permitted. The original Criteo preprocessing provenance limitation remains.

| Dataset | Train rows | Validation rows | Test rows | Fields | Embedding dimension |
| --- | ---: | ---: | ---: | ---: | ---: |
| Criteo | 36,672,493 | 4,584,062 | 4,584,062 | 39 | 16 |
| Avazu | 32,343,172 | 4,042,897 | 4,042,898 | 24 | 16 |

Validate metadata, array shape and full cache-file hashes before training. Use MPS float32, PyTorch intra-op threads 4 and inter-op threads 1, with no CPU fallback. The released trainer is unchanged. Adam LR is 0.0003, embedding LR multiplier 1, batch 4096, embedding initialization SD 0.0001, embedding regularization 0, gradient clip 10, train_frac 1, no scheduler, and no added regularization/dropout. Diagnostics erank/funnel/agop are disabled. Every training row, including the final short batch, is used in each epoch.

The released runner's seed*1000+epoch minibatch permutation is retained, giving the same order for a dataset/seed/epoch across arms. Different model constructions consume random numbers differently; identical seed numbers do not imply bitwise identical embeddings or shared-projection initialization.

After each epoch evaluate the full validation split. Update the selected checkpoint only if validation AUC exceeds the previous best by more than 1e-6; otherwise keep the earlier epoch. Restore that checkpoint and evaluate the full test split exactly once. Do not select by test metrics or LogLoss, and do not force epoch 2. The runner may reduce LR after the second validation, but no subsequent training occurs, so both training passes retain LR 0.0003. Preserve both epoch LR records.

The trainer may automatically write test metrics at a run's end. Humans and analysis tools must not inspect metric files until BOTH separately frozen queues have completed: the original 12 plus the secondary native-field 6, for 18 total. The secondary plan was frozen after these initial 12 began, before human test inspection; do not describe all 18 as jointly frozen before the first fit. Status/progress used for monitoring must exclude validation/test metric values.

## Fixed analysis and interpretation

Primary: for every dataset/seed, delta AUC = AUC(selection) - AUC(learned_budget). Report all six paired differences, and each dataset's mean and sample SD of three differences, plus each arm's mean and sample SD. Do not pool datasets into a primary estimate or combine these runs with historical runs.

Secondary: delta LogLoss = LogLoss(selection) - LogLoss(learned_budget), at the same selected checkpoint. Negative favors selection. Also disclose every epoch's validation AUC/LogLoss, selected epoch, actual training rows/steps, runtime, failures/retries, parameter counts, matrix FLOPs and cost ratios. Matrix FLOPs are twice nn.Linear MACs for these two pml_mv arms, which have no additional non-Linear matrix products.

Use descriptive analysis only: no p-values, general superiority, equivalence/noninferiority conclusion, or post-hoc practical-equivalence margin. Sample SD reflects seed variation on fixed splits/configurations, not new evaluation datasets.

If both dataset mean AUC differences are positive, describe preservation of mean ordering under this one alternative protocol. Say every new seed has the favorable direction only if all six AUC differences are positive. Report reversals, mixed signs and AUC/LogLoss tradeoffs. Validation gains/epoch-2 selections may be compatible with different early optimization speed; they do not establish a causal decomposition or shrinkage of an old gap. Rising epoch-2 validation is not proof of convergence. Do not tune LR, width or seeds to replace the result.

Report all failures. Numerical failure is an outcome; documented infrastructure failure may be retried only at the identical frozen configuration with a separate record. Do not report an incomplete plan as complete. The supplied launchers do not retry automatically. Keep the frozen files unchanged and record deviations separately.
