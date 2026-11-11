"""MaskNet and its pyramidal conversion (Table 2).

    MaskNet          hand port of FuxiCTR v2.2.0 model_zoo/MaskNet (ParallelMaskNet)
    PyramidMaskNet   the same mask blocks followed by a DNN whose widths do not increase

The official Criteo_x4 configuration has two parallel blocks of width 500, reduction ratio 0.1, a 5 x 1000 DNN,
dropout 0.3 and no layer normalization.  The released configurations keep the number of blocks and the
normalization settings and set the widths, the dropout and the reduction ratio (configs/).
"""
import torch
import torch.nn as nn

from .base import BaseModel
from .layers import MLP_Block, get_activation


class MaskBlock(nn.Module):
    """MaskNet.py MaskBlock: LN?(W_h (mask(V_emb) * V_hidden)) -> act -> dropout,
    mask = W2 ReLU(W1 V_emb) with W1: input_dim -> int(hidden_dim * reduction_ratio)."""

    def __init__(self, input_dim, hidden_dim, output_dim, hidden_activation="ReLU", reduction_ratio=1,
                 dropout_rate=0, layer_norm=True):
        super().__init__()
        self.mask_layer = nn.Sequential(nn.Linear(input_dim, int(hidden_dim * reduction_ratio)),
                                        nn.ReLU(),
                                        nn.Linear(int(hidden_dim * reduction_ratio), hidden_dim))
        hidden_layers = [nn.Linear(hidden_dim, output_dim, bias=False)]
        if layer_norm:
            hidden_layers.append(nn.LayerNorm(output_dim))
        hidden_layers.append(get_activation(hidden_activation))
        if dropout_rate > 0:
            hidden_layers.append(nn.Dropout(p=dropout_rate))
        self.hidden_layer = nn.Sequential(*hidden_layers)

    def forward(self, V_emb, V_hidden):
        return self.hidden_layer(self.mask_layer(V_emb) * V_hidden)


class ParallelMaskNet(nn.Module):
    """MaskNet.py ParallelMaskNet: parallel mask blocks followed by a DNN."""

    def __init__(self, input_dim, output_dim=None, num_blocks=1, block_dim=64, hidden_units=(),
                 hidden_activations="ReLU", reduction_ratio=1, dropout_rates=0, layer_norm=True):
        super().__init__()
        self.num_blocks = num_blocks
        self.mask_blocks = nn.ModuleList(
            MaskBlock(input_dim, input_dim, block_dim, hidden_activations, reduction_ratio, dropout_rates,
                      layer_norm) for _ in range(num_blocks))
        # official output_activation (Sigmoid) folded into the loss
        self.dnn = MLP_Block(input_dim=block_dim * num_blocks, output_dim=output_dim, hidden_units=hidden_units,
                             hidden_activations=hidden_activations, output_activation=None,
                             dropout_rates=dropout_rates)

    def forward(self, V_emb, V_hidden):
        return self.dnn(torch.cat([blk(V_emb, V_hidden) for blk in self.mask_blocks], dim=-1))


class MaskNet(BaseModel):
    """model_zoo/MaskNet/src/MaskNet.py @FuxiCTR v2.2.0 (ParallelMaskNet)."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, dnn_hidden_units=(64, 64, 64),
                 dnn_hidden_activations="ReLU", model_type="ParallelMaskNet", parallel_num_blocks=1,
                 parallel_block_dim=64, reduction_ratio=1, net_dropout=0, emb_layernorm=True, net_layernorm=True):
        super().__init__(vocab_size, num_fields, emb_dim)
        assert model_type == "ParallelMaskNet", model_type
        input_dim = num_fields * emb_dim
        self.mask_net = ParallelMaskNet(input_dim, 1, parallel_num_blocks, parallel_block_dim,
                                        list(dnn_hidden_units), dnn_hidden_activations, reduction_ratio,
                                        net_dropout, net_layernorm)
        self.emb_norm = nn.ModuleList(nn.LayerNorm(emb_dim) for _ in range(num_fields)) if emb_layernorm else None
        self.reset_parameters()

    def logits(self, feature_emb):
        if self.emb_norm is not None:
            feat_list = feature_emb.chunk(self.num_fields, dim=1)
            V_hidden = torch.cat([self.emb_norm[i](f) for i, f in enumerate(feat_list)], dim=1)
        else:
            V_hidden = feature_emb
        return self.mask_net(feature_emb.flatten(start_dim=1), V_hidden.flatten(start_dim=1)).squeeze(-1)


class PyramidParallelMaskNet(nn.Module):
    """Parallel mask blocks followed by a DNN with the given widths."""

    def __init__(self, blocks, dnn_input, dnn_widths, activations, dropout):
        super().__init__()
        self.mask_blocks = nn.ModuleList(blocks)
        self.dnn = MLP_Block(dnn_input, dnn_widths, activations, 1, None, dropout)  # official Sigmoid -> loss

    def forward(self, V_emb, V_hidden):
        return self.dnn(torch.cat([blk(V_emb, V_hidden) for blk in self.mask_blocks], dim=-1))


class PyramidMaskNet(BaseModel):
    """ParallelMaskNet (official MaskBlock, input-space instance mask) with a non-increasing DNN."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, block_dim=312, dnn_widths=(624,) * 5, num_blocks=2,
                 reduction_ratio=1.0, net_dropout=0.3, activations="relu", net_layernorm=False):
        super().__init__(vocab_size, num_fields, emb_dim)
        d0 = num_fields * emb_dim
        self.d0, self.block_dim, self.num_blocks, self.dnn_widths = d0, block_dim, num_blocks, list(dnn_widths)
        self.reduction_ratio = reduction_ratio
        blocks = [MaskBlock(d0, d0, block_dim, activations, reduction_ratio, net_dropout, net_layernorm)
                  for _ in range(num_blocks)]
        self.mask_net = PyramidParallelMaskNet(blocks, num_blocks * block_dim, self.dnn_widths, activations,
                                               net_dropout)
        self.reset_parameters()

    def logits(self, feature_emb):
        V = feature_emb.flatten(start_dim=1)  # emb_layernorm=False (official): V_hidden = V_emb
        return self.mask_net(V, V).squeeze(-1)
