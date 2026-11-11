"""FinalMLP and its pyramidal conversion (Table 2).

    FinalMLP          hand port of FuxiCTR v2.2.0 model_zoo/FinalMLP (without feature selection)
    PyramidFinalMLP   the two streams with per-layer widths

The official Criteo_x4 configuration has a 5 x 1000 stream (dropout 0.2), a 1 x 500 stream (dropout 0.1), no batch
normalization and 50 heads.  The released configurations set the widths, the number of heads and the dropout
(configs/).
"""
import torch
import torch.nn as nn

from .base import BaseModel
from .layers import MLP_Block, make_mlp


class InteractionAggregation(nn.Module):
    """FinalMLP.py multi-head bilinear fusion: w_x x + w_y y + sum_h x_h^T W_h y_h."""

    def __init__(self, x_dim, y_dim, output_dim=1, num_heads=1):
        super().__init__()
        assert x_dim % num_heads == 0 and y_dim % num_heads == 0
        self.num_heads, self.output_dim = num_heads, output_dim
        self.head_x_dim, self.head_y_dim = x_dim // num_heads, y_dim // num_heads
        self.w_x = nn.Linear(x_dim, output_dim)
        self.w_y = nn.Linear(y_dim, output_dim)
        self.w_xy = nn.Parameter(torch.Tensor(num_heads * self.head_x_dim * self.head_y_dim, output_dim))
        nn.init.xavier_normal_(self.w_xy)

    def forward(self, x, y):
        output = self.w_x(x) + self.w_y(y)
        head_x = x.view(-1, self.num_heads, self.head_x_dim)
        head_y = y.view(-1, self.num_heads, self.head_y_dim)
        xy = torch.matmul(torch.matmul(head_x.unsqueeze(2), self.w_xy.view(self.num_heads, self.head_x_dim, -1))
                          .view(-1, self.num_heads, self.output_dim, self.head_y_dim),
                          head_y.unsqueeze(-1)).squeeze(-1)
        return output + xy.sum(dim=1)


class FinalMLP(BaseModel):
    """model_zoo/FinalMLP/src/FinalMLP.py @FuxiCTR v2.2.0 without feature selection (use_fs=False, the official
    Criteo_x4 setting)."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, mlp1_hidden_units=(64, 64, 64),
                 mlp1_hidden_activations="ReLU", mlp1_dropout=0, mlp1_batch_norm=False,
                 mlp2_hidden_units=(64, 64, 64), mlp2_hidden_activations="ReLU", mlp2_dropout=0,
                 mlp2_batch_norm=False, num_heads=1):
        super().__init__(vocab_size, num_fields, emb_dim)
        feature_dim = emb_dim * num_fields
        mlp1_hidden_units, mlp2_hidden_units = list(mlp1_hidden_units), list(mlp2_hidden_units)
        self.mlp1 = MLP_Block(feature_dim, mlp1_hidden_units, mlp1_hidden_activations, None, None, mlp1_dropout,
                              mlp1_batch_norm)
        self.mlp2 = MLP_Block(feature_dim, mlp2_hidden_units, mlp2_hidden_activations, None, None, mlp2_dropout,
                              mlp2_batch_norm)
        self.fusion_module = InteractionAggregation(mlp1_hidden_units[-1], mlp2_hidden_units[-1], 1, num_heads)
        self.reset_parameters()

    def logits(self, feature_emb):
        flat_emb = feature_emb.flatten(start_dim=1)
        return self.fusion_module(self.mlp1(flat_emb), self.mlp2(flat_emb)).squeeze(-1)


class NarrowingInteractionAggregation(InteractionAggregation):
    """FinalMLP bilinear fusion with identical parameters/function; the per-head bilinear x_h^T W_h y_h is
    contracted from the wider head vector into the narrower one (official order x-first when dx >= dy), so the
    intermediate matmul never up-projects.  output_dim must be 1 (official)."""

    def __init__(self, x_dim, y_dim, output_dim=1, num_heads=1):
        assert output_dim == 1
        super().__init__(x_dim, y_dim, output_dim, num_heads)

    def forward(self, x, y):
        if self.head_y_dim <= self.head_x_dim:
            return super().forward(x, y)
        output = self.w_x(x) + self.w_y(y)
        H, dx, dy = self.num_heads, self.head_x_dim, self.head_y_dim
        head_x = x.view(-1, H, dx)
        head_y = y.view(-1, H, dy)
        Wy = torch.matmul(self.w_xy.view(H, dx, dy), head_y.unsqueeze(-1))  # B x H x dx x 1
        xy = torch.matmul(head_x.unsqueeze(2), Wy).view(-1, H, 1)  # B x H x 1
        return output + xy.sum(dim=1)


class PyramidFinalMLP(BaseModel):
    """FinalMLP (no feature selection, as the official Criteo_x4 config) with pyramid stream widths."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, mlp1_widths=(624,) * 5, mlp2_widths=(500,),
                 num_heads=8, mlp1_dropout=0.2, mlp2_dropout=0.1, activations="relu", batch_norm=False, res="state",
                 rr="slice"):
        super().__init__(vocab_size, num_fields, emb_dim)
        d0 = num_fields * emb_dim
        self.d0, self.mlp1_widths, self.mlp2_widths = d0, list(mlp1_widths), list(mlp2_widths)
        self.num_heads = num_heads
        self.mlp1 = make_mlp(d0, self.mlp1_widths, activations, mlp1_dropout, batch_norm, res, rr)
        self.mlp2 = make_mlp(d0, self.mlp2_widths, activations, mlp2_dropout, batch_norm, res, rr)
        self.fusion_module = NarrowingInteractionAggregation(self.mlp1_widths[-1], self.mlp2_widths[-1], 1,
                                                             self.num_heads)
        self.reset_parameters()

    def logits(self, feature_emb):
        flat = feature_emb.flatten(start_dim=1)
        return self.fusion_module(self.mlp1(flat), self.mlp2(flat)).squeeze(-1)
