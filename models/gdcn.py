"""GDCN on the shared projected input, and its pyramidal conversion (Table 2).

Gated cross layers next to a parallel DNN.  With constant cross_widths the cross layer is GDCN's gated cross layer
(Table 1); narrowing cross_widths and dnn_hidden_units give the pyramidal conversion.
"""
import torch
import torch.nn as nn

from .base import BaseModel
from .layers import even_idx, make_mlp, make_shrink


class GatedCrossNetwork(nn.Module):
    """GDCN gated cross layer generalized to d_{i-1} -> d_i:
    x_i = P_i(x0) * (W_c x_{i-1} + b) * sigmoid(W_g x_{i-1}) + S_i(x_{i-1}),  W_c, W_g: d_{i-1} -> d_i.
    With equal widths this is GDCN's GateCrossLayer."""

    def __init__(self, d_in, widths, shrink="slice", res="state"):
        super().__init__()
        self.res = res  # "reread": the residual is the anchor P_i(x0) instead of S_i(x_{i-1})
        self.w, self.wg, self.b = nn.ModuleList(), nn.ModuleList(), nn.ParameterList()
        self.skip, self.anchor = nn.ModuleList(), nn.ModuleList()
        prev, cum = d_in, torch.arange(d_in)
        for i, d in enumerate(widths):
            assert d <= prev or (i == 0 and shrink == "slice")  # only the first layer may widen (input-side copy)
            self.w.append(nn.Linear(prev, d, bias=False))
            self.wg.append(nn.Linear(prev, d, bias=False))
            self.b.append(nn.Parameter(torch.zeros((d,))))
            nn.init.uniform_(self.b[i].data)
            self.skip.append(make_shrink(prev, d, shrink))
            cum = cum[even_idx(prev, d)] if d != prev else cum
            self.anchor.append(make_shrink(d_in, d, shrink, idx=cum.clone() if (shrink == "slice" and d != d_in)
                                              else None))
            prev = d
        self.out_dim = prev

    def forward(self, x):
        x0 = x
        for i in range(len(self.w)):
            a = self.anchor[i](x0)
            c = self.w[i](x) + self.b[i]
            x = a * c * torch.sigmoid(self.wg[i](x)) + (self.skip[i](x) if self.res == "state" else a)
        return x


class GDCN(BaseModel):
    """Gated cross layers with widths cross_widths and a parallel DNN with widths dnn_hidden_units; d is the
    width of the projected input."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, d=256, cross_widths=None, dnn_hidden_units=None,
                 dnn_activations="ReLU", net_dropout=0.5, batch_norm=True, shrink="slice", res="state", rr="slice"):
        super().__init__(vocab_size, num_fields, emb_dim)
        self.d0, self.d = num_fields * emb_dim, d
        assert d == self.d0  # reads the projected input directly
        self.cross_widths = list(cross_widths)
        self.dnn_widths = list(dnn_hidden_units)
        self.cross_net = GatedCrossNetwork(d, self.cross_widths, shrink, res)
        self.dnn = make_mlp(d, self.dnn_widths, dnn_activations, net_dropout, batch_norm, res, rr)
        self.fc = nn.Linear(self.cross_widths[-1] + self.dnn_widths[-1], 1)
        self.reset_parameters()

    def logits(self, feature_emb):
        x0 = feature_emb.flatten(start_dim=1)
        return self.fc(torch.cat([self.cross_net(x0), self.dnn(x0)], dim=1)).squeeze(-1)
