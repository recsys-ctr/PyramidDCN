"""Building blocks shared by the models.

    get_activation, MLP_Block, multihead_feature_emb   FuxiCTR v2.2.0 pieces used by the baselines
    StridedSlice, even_idx                             parameter-free channel selection
    make_shrink, make_reread, ReReadMLP, make_mlp      width changes of the pyramidal conversions (Table 2)

Width changes on residual / anchor paths use parameter-free evenly spaced coordinate selections
idx_j = floor(j * d_in / d_out) (== x[..., ::2] for halving); for anchors the indices are composed through the
layers so that the anchor and the state shortcut select the same input coordinates.
"""
import torch
import torch.nn as nn


def get_activation(name):
    """fuxictr.pytorch.torch_utils.get_activation (string branch)."""
    if not isinstance(name, str):
        return name
    n = name.lower()
    if n == "relu":
        return nn.ReLU()
    if n == "sigmoid":
        return nn.Sigmoid()
    if n == "tanh":
        return nn.Tanh()
    if n == "softmax":
        return nn.Softmax(dim=-1)
    return getattr(nn, name)()


class MLP_Block(nn.Module):
    """fuxictr/pytorch/layers/blocks/mlp_block.py @v2.2.0 (bn_only_once=False):
    [Linear -> (BatchNorm1d) -> act -> (Dropout)] x len(hidden_units) -> (Linear out) -> (out act)."""

    def __init__(self, input_dim, hidden_units=(), hidden_activations="ReLU", output_dim=None,
                 output_activation=None, dropout_rates=0.0, batch_norm=False, use_bias=True):
        super().__init__()
        hidden_units = list(hidden_units)
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        dims = [input_dim] + hidden_units
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=use_bias))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            if hidden_activations[i]:
                layers.append(get_activation(hidden_activations[i]))
            if dropout_rates[i] > 0:
                layers.append(nn.Dropout(p=dropout_rates[i]))
        if output_dim is not None:
            layers.append(nn.Linear(dims[-1], output_dim, bias=use_bias))
        if output_activation is not None:
            layers.append(get_activation(output_activation))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


def multihead_feature_emb(feature_emb, num_heads):
    """MultiHeadFeatureEmbedding.forward minus the lookup: B x F x (D*H) -> B x H x (F*D).
    Per head: [first half of every field's dims (field-major), then second half of every field's dims]."""
    mh = torch.stack(torch.tensor_split(feature_emb, num_heads, dim=-1), dim=1)  # B x H x F x D
    e1, e2 = torch.tensor_split(mh, 2, dim=-1)  # B x H x F x D/2 each
    return torch.cat([e1.flatten(start_dim=2), e2.flatten(start_dim=2)], dim=-1)


class StridedSlice(nn.Module):
    """Parameter-free coordinate selection (non-persistent index buffer)."""

    def __init__(self, idx):
        super().__init__()
        self.register_buffer("idx", torch.as_tensor(idx, dtype=torch.long), persistent=False)

    def forward(self, x):
        return x.index_select(-1, self.idx)


def even_idx(d_in, d_out):
    return (torch.arange(d_out) * d_in) // d_out


def make_shrink(d_in, d_out, mode="slice", idx=None):
    if d_out == d_in and idx is None:
        return nn.Identity()
    assert mode == "slice", mode
    return StridedSlice(even_idx(d_in, d_out) if idx is None else idx)  # may widen the input by copying coordinates


def make_reread(d_src, widths, rr="slice"):
    """Re-read residuals r_i: d_src -> d_i (strided slice; Identity when d_i == d_src)."""
    return nn.ModuleList(make_shrink(d_src, d, rr) for d in widths)


class ReReadMLP(MLP_Block):
    """MLP_Block whose hidden layer i adds a re-read of the block input: h_i = block_i(h_{i-1}) + r_i(x)
    (the residual of the narrowing design applied to an MLP; widths must not exceed the input width).
    Same `mlp` Sequential as MLP_Block, so official init code that walks `mlp` still applies."""

    def __init__(self, input_dim, hidden_units=(), hidden_activations="ReLU", output_dim=None, output_activation=None,
                 dropout_rates=0.0, batch_norm=False, use_bias=True, rr="slice"):
        super().__init__(input_dim, hidden_units, hidden_activations, output_dim, output_activation, dropout_rates,
                         batch_norm, use_bias)
        assert output_dim is None and output_activation is None
        starts = [i for i, m in enumerate(self.mlp) if isinstance(m, nn.Linear)] + [len(self.mlp)]
        self.blocks = list(zip(starts[:-1], starts[1:]))
        self.reread = make_reread(input_dim, list(hidden_units), rr)

    def forward(self, x):
        h = x
        for (a, b), r in zip(self.blocks, self.reread):
            for m in self.mlp[a:b]:
                h = m(h)
            h = h + r(x)
        return h


def make_mlp(input_dim, hidden_units, act, dropout, batch_norm, res="state", rr="slice"):
    if res == "reread":
        return ReReadMLP(input_dim, hidden_units, act, None, None, dropout, batch_norm, rr=rr)
    return MLP_Block(input_dim, hidden_units, act, None, None, dropout, batch_norm)
