# Selected Table 2 architectures

Original, **S** (same initial width), and **B** (near-matched matrix FLOPs)
refer to the three reported conditions. Width lists are outputs of successive
layers; `||` denotes parallel paths. Every condition uses seeds 2024–2026 and
one epoch. All constructor arguments are in `configs/table2/criteo.json`, with
a compact export in `configs/table2_architectures.csv`.

Conversions preserve defining operators where feasible, narrow constant-width
cross/MLP layers, and use fixed input selection where compatible. They are
model-specific adaptations, not a uniform intervention that changes only
width. The selected earlier QNN-α and FinalMLP budget variants also change
depth and retain an initial expansion. These exceptions are part of the
reported experiment and must not be silently replaced when reproducing it.

| Model | Original | S | B |
| --- | --- | --- | --- |
| DCNv2 | input 512; cross `[512,512,512]` || MLP `[512,512,512]` | input 512; both `[512,256,128]` | input 672; both `[672,336,168]` |
| ML-DCN | input 512; cross `[512,512,512]` | input 512; cross `[512,256,128]` | input 624; cross `[624,312,156]` |
| MaskNet | input 512; serial blocks `[512,512,512]`; scalar head | input 512; blocks `[512,256,128]`; scalar head | input 960; blocks `[960,480,240]`; scalar head |
| QNN-α | input 512; quadratic `[512,512,512,512]` | input 512; quadratic `[512,256,128,64]` | input 512; quadratic `[944,472,236]` |
| FCN_p | input 512; ECN `[512,512,512]` || LCN `[512,512,512]` | input 512; both `[512,256,128]` | input 648; both `[648,324,162]` |
| GDCN | input 512; gated cross `[512,512,512]` || MLP `[512,512,512]` | input 512; both `[512,256,128]` | input 720; cross `[720,360,180]` || MLP `[512,256,128]` |
| FinalMLP | input 512; MLP1 `[512,512,512,512,512]` || MLP2 `[512]` | input 512; MLP1 `[512,256,128,64,32]` || MLP2 `[512]` | input 512; MLP1 `[1096,544,272]` || MLP2 `[512]` |
| FiBiNet++ | input 512; fixed interaction front end; MLP `[512,512,512]` | same front end; MLP `[512,256,128]` | same front end; MLP `[768,384,192]` |

## Cross operators and connections

- **DCNv2:** dense full matrices (no low-rank bottleneck), three cross layers
  and a parallel three-layer ReLU/BatchNorm MLP. Each cross multiplies its
  state transformation by the projected input or its balanced fixed slice.
  Preserve the previous-state residual; slice it when dimensions differ.
- **ML-DCN:** retain both V and U around masking. Original uses rank 512 and
  mask paths `512 → 256 → 512`. In a converted layer of width d, the cross path
  is `previous → d → d` and the mask path is `previous → d → d`, with ReLU
  between mask maps. Use balanced input slices, an evenly spaced previous-state
  residual, and LayerNorm. The last mask map starts at weight zero and bias one.
- **MaskNet:** three serial MaskBlocks, without a prediction MLP. Original
  masks follow `512 → 1024 → 512`; each masked state is linearly transformed,
  normalized and passed through ReLU. Converted masks follow `d → d → d` and
  read a balanced input slice. Before multiplication, contiguous groups of
  previous-state channels are summed and divided by the square root of group
  size. The product passes through `Linear(d,d)`, LayerNorm and ReLU. No additive
  input skip is introduced. No embedding LayerNorm is used.
- **QNN-α:** five parallel projection rows per quadratic layer, one head,
  no dropout or BatchNorm. Original uses the previous state as multiplier and
  residual. Converted variants use fixed projected-input selections for both.
  B has three layers instead of four and its first per-row map expands
  `512 → 944`; selections repeat coordinates when necessary.
- **FCN_p:** three ECN layers and three LCN layers in parallel, with mean-logit
  fusion and Tri-BCE. Each LCA layer linearly produces d/2 channels and
  concatenates them with an affine copy to obtain d channels. ECN multiplies
  by the previous state; LCN by the projected input. Converted LCN uses balanced
  input slices; previous-state residuals are sliced. No BatchNorm, LayerNorm
  or dropout. ECN is an internal branch, not a separate table baseline.
- **GDCN:** preserve sigmoid-gated crosses, previous-state residuals and the
  parallel ReLU/BatchNorm MLP. Converted crosses read balanced input slices;
  residuals use fixed previous-state slices. B retains MLP widths
  `[512,256,128]` while enlarging the input and cross path to 720.
- **FinalMLP:** retain two MLP streams and eight-head bilinear fusion; feature
  selection is disabled in all three selected configurations. Converted
  streams add fixed input selections after hidden blocks. B reduces MLP1
  depth from five to three and retains the `512 → 1096` expansion. Its final
  bilinear contraction computes the same fusion with the narrower output
  dimension as the intermediate. No new cross stack is introduced.
- **FiBiNet++:** retain 64 projected field groups of eight channels, bilinear
  interaction rank 50, and bit-wise SENet. SENet retains `256 → 85 → 512`.
  Only the three-layer prediction MLP narrows. Its input width is 562, so B
  retains the initial `562 → 768` expansion. The interaction front end is
  unchanged.

The Table 1 Criteo baseline configurations and results are the same Original
runs. Other Table 1 datasets use the corresponding projected-input width;
see their JSON files for the exact field counts and model dimensions.
