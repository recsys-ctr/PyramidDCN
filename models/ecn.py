"""ECN and its pyramidal conversion (Table 2).

    ECN          exponential cross network with a self-mask (official Criteo configuration); the Table 1
                 configurations use constant widths
    PyramidECN   narrowing widths; every layer re-reads the input
"""
import torch
import torch.nn as nn

from .base import BaseModel
from .layers import make_reread, make_shrink, multihead_feature_emb


class ExponentialCrossNetwork(nn.Module):
    """FCN@master ECN layer generalized to d_in -> d_out (even): H = W x (d_in -> d_out/2, no bias);
    z = [H, H * ReLU(LN(H))] (d_out); x' = S(x) * (z + b) + S(x); dropout.  widths == [d0]*L is the official ECN."""

    def __init__(self, d0, widths, layer_norm=True, batch_norm=False, net_dropout=0.1, num_heads=1, shrink="slice",
                 res="state", rr="slice"):
        super().__init__()
        self.reread = make_reread(d0, widths, rr) if res in ("reread", "reread_anchor") else None  # residual r_i(x_in)
        self.anchor_reread = res == "reread_anchor"  # r_i(x_in) is also the multiplier instead of S(x)
        self.layer_norm, self.batch_norm, self.dropout = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.w, self.b, self.skip = nn.ModuleList(), nn.ParameterList(), nn.ModuleList()
        prev = d0
        for i, d in enumerate(widths):
            assert d % 2 == 0 and (d <= prev or (i == 0 and shrink == "slice")), (prev, d)  # first layer may copy-widen
            self.w.append(nn.Linear(prev, d // 2, bias=False))
            self.b.append(nn.Parameter(torch.empty((d,))))
            if layer_norm:
                self.layer_norm.append(nn.LayerNorm(d // 2))
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(num_heads))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))
            nn.init.uniform_(self.b[i].data)
            self.skip.append(make_shrink(prev, d, shrink))
            prev = d
        self.masker = nn.ReLU()
        self.fc = nn.Linear(prev, 1)

    def forward(self, x):
        x_in = x
        for i in range(len(self.w)):
            H = self.w[i](x)
            if len(self.batch_norm) > i:
                H = self.batch_norm[i](H)
            mask = self.masker(self.layer_norm[i](H)) if len(self.layer_norm) > i else self.masker(H)
            z = torch.cat([H, H * mask], dim=-1)
            s = self.reread[i](x_in) if self.anchor_reread else self.skip[i](x)  # S unused with anchor re-read
            zb = z + self.b[i]
            x = s * zb + (s if self.reread is None else self.reread[i](x_in))
            if len(self.dropout) > i:
                x = self.dropout[i](x)
        return self.fc(x)


class ECN(BaseModel):
    """ECN (self-mask, official Criteo config: LN True, BN False, 1 head, plain BCE)."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, widths=(624,) * 4, net_dropout=0.1, layer_norm=True,
                 batch_norm=False, shrink="slice"):
        super().__init__(vocab_size, num_fields, emb_dim)
        d0 = num_fields * emb_dim
        self.d0, self.widths = d0, list(widths)
        self.ECN = ExponentialCrossNetwork(d0, self.widths, layer_norm, batch_norm, net_dropout, 1, shrink)
        self.reset_parameters()

    def logits(self, feature_emb):
        return self.ECN(multihead_feature_emb(feature_emb, 1)).mean(dim=1).squeeze(-1)


class PyramidECN(BaseModel):
    """ECN with narrowing widths (Table 2).  The first layer keeps the width d of the projected input; later
    layers narrow.  With res="reread_anchor" every layer re-reads the input through a parameter-free strided
    slice, as the multiplier and as the skip term."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, d=256, widths=(256, 128, 64, 32), net_dropout=0.1,
                 shrink="slice", res="state", rr="slice"):
        super().__init__(vocab_size, num_fields, emb_dim)
        self.d0, self.d, self.widths = num_fields * emb_dim, d, list(widths)
        assert d == self.d0  # reads the projected input directly
        self.ECN = ExponentialCrossNetwork(d, self.widths, True, False, net_dropout, 1, shrink, res, rr)
        self.reset_parameters()

    def logits(self, feature_emb):
        return self.ECN(multihead_feature_emb(feature_emb, 1)).mean(dim=1).squeeze(-1)
