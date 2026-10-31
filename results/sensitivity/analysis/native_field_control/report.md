# Native-field FiBiNet++ comparison under the added protocol

Both separately frozen queues, all eighteen fits, completed before analysis.

| Dataset | Seed | Delta AUC (Pyramid - native FiBiNet++) | Delta LogLoss |
|---|---:|---:|---:|
| criteo | 2032 | -0.000971108727825 | 0.0010363363403 |
| criteo | 2033 | -0.000713521168844 | 0.000533215663319 |
| criteo | 2034 | -0.000693436505118 | 0.000730737535417 |
| avazu | 2032 | 0.00365891904668 | -0.00236563834005 |
| avazu | 2033 | 0.00362381073622 | -0.00263752805931 |
| avazu | 2034 | 0.00336961096554 | -0.0021492802087 |

| Dataset | Mean delta AUC | Sample SD | Mean delta LogLoss | Sample SD |
|---|---:|---:|---:|---:|
| criteo | -0.000792688800596 | 0.000154842181765 | 0.000766763179679 | 0.000253487650952 |
| avazu | 0.00355078024948 | 0.00015787615541 | -0.00238414886935 | 0.000244649689715 |

Report all seed and metric directions. This comparison does not identify a pure projection-removal effect or establish a native-input SOTA reproduction.

- Native-field FiBi++ comparison under the added protocol, not a pure projection-removal causal ablation.
- No projected-FiBi++ run under the added protocol; old Table1 differs in LR, epoch, device, and seeds.
- Exact Table1 hand-port DNN overrides retained; not an independently tuned official reproduction or native-input SOTA test.
- Different input grouping and parameter/FLOP budgets; descriptive paired AUC/LogLoss only, no p-values or general superiority.
- The same six new PyramidDCN results are reused across the two separate comparisons; they are not independent replications.
- Three seeds per fixed benchmark split; two-epoch cap, reused test splits, and no convergence claim.
