# Fixed alternative-protocol sensitivity analysis

Both queues passed the eighteen-fit guard; initial twelve fits completed without failure. Analysis uses full-precision full-test evaluation returns.

Delta is selection minus learned_budget; positive AUC and negative LogLoss favor selection.

| Dataset | Seed | Delta AUC | Delta LogLoss | Selection/learned FLOPs |
|---|---:|---:|---:|---:|
| criteo | 2032 | 0.000432798567618 | -0.000431942571423 | 1.0146619744 |
| criteo | 2033 | 0.000561875068623 | -0.000472167845979 | 1.0146619744 |
| criteo | 2034 | 0.000903592951435 | -0.000780152664328 | 1.0146619744 |
| avazu | 2032 | 0.00130656319322 | -0.000801183869949 | 1.01809408043 |
| avazu | 2033 | 0.00107521089837 | -0.000660715427793 | 1.01809408043 |
| avazu | 2034 | 0.000823046853397 | -0.000497399712431 | 1.01809408043 |

| Dataset | Mean delta AUC | Sample SD | Mean delta LogLoss | Sample SD |
|---|---:|---:|---:|---:|
| criteo | 0.000632755529225 | 0.000243269126133 | -0.000561421027243 | 0.000190491902648 |
| avazu | 0.00106827364833 | 0.000241832807531 | -0.000653099670057 | 0.000152035204522 |

Mean AUC ordering is preserved on both datasets under this one fixed alternative protocol.
All six AUC differences are positive, so every new seed has the same observed favorable AUC direction.
Interpret difference sizes together with sample SD; mean ordering does not establish general superiority or statistical significance.
Report validation AUC changes and epoch-2 selection counts by arm. They may be compatible with different early optimization speed but do not separate LR/epoch causal effects or establish shrinkage of an old gap.
Results are conditional on a two-epoch cap. Report epoch-2 validation change without claiming convergence.

All twelve runs, per-epoch validation, selected epochs, rows/steps, runtime, parameters and FLOPs are retained in runs.csv and summary.json.

- One prespecified alternative protocol: fixed Adam LR 3e-4, two full epochs, validation AUC checkpoint selection.
- Datasets are analyzed separately; no pooled primary estimate, significance test, p-value, superiority, equivalence, or noninferiority claim.
- Three paired seeds per fixed split/configuration; sample SD describes only this seed variation.
- Existing benchmark test splits are reused; new seeds are not new evaluation data.
- Near-matched matrix-FLOP recipe comparison with different state widths; not a pure input-map causal effect.
- LR and checkpoint/epoch effects are not separately identified; no convergence or native-input SOTA conclusion.
- No historical 798-run pooling, historical metric replacement, result-dependent tuning, or new practical-equivalence margin.
- Restoration of checkpoint weights relies on the released runner; this analysis verifies recorded best_epoch and evaluation metadata, not saved weights.
- Original rounded history times/losses retain their recorded precision; full-precision evaluation returns are used for metric analysis.
