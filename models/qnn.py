"""QNN-alpha and its pyramidal conversion (Table 2).

    QNN          QNN-alpha (salmon1802/QNN src/QNN_T26.py) with per-layer rows and the official SE loss; the
                 Table 1 configurations use constant widths
    PyramidQNN   narrowing widths; every layer re-reads the input
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel
from .layers import make_reread, make_shrink


class QuadraticLayer(nn.Module):
    """QNN_T26 QuadraticLayer (num_heads=1) generalized to d_in -> d_out with R rows:
    h = Dropout(ReLU(BN?(W x)).view(R, d_out)); out = S(x) * sum_r h_r + S(x)."""

    def __init__(self, d_in, d_out, num_row, batch_norm=False, net_dropout=0.1, shrink="slice", reread=None,
                 anchor_reread=False):
        super().__init__()
        self.reread = reread  # if set, the residual is reread(x0) instead of S(x)
        self.anchor_reread = anchor_reread  # if set, reread(x0) is also the multiplier (anchor) instead of S(x)
        self.linear = nn.Sequential(nn.Linear(d_in, d_out * num_row),
                                    nn.BatchNorm1d(d_out * num_row) if batch_norm else nn.Identity(),
                                    nn.ReLU())
        if net_dropout > 0:
            self.dropout = nn.Dropout(net_dropout)
        self.net_dropout, self.num_row, self.d_out = net_dropout, num_row, d_out
        self.skip = make_shrink(d_in, d_out, shrink)

    def forward(self, x, x0=None):
        ego_x = self.reread(x0) if self.anchor_reread else self.skip(x)  # the state shortcut S is unused then
        h = self.linear(x).view(-1, self.num_row, self.d_out)
        if self.net_dropout > 0:
            h = self.dropout(h)
        return (ego_x.unsqueeze(1) * h).sum(dim=1) + (ego_x if self.reread is None else self.reread(x0))


class QuadraticNetwork(nn.Module):
    """Stack of QuadraticLayer (rows[i] rows in layer i) and a linear head."""

    def __init__(self, d0, widths, rows, net_dropout, batch_norm, shrink, res="state", rr="slice"):
        super().__init__()
        self.layer, self.dropout = nn.ModuleList(), nn.ModuleList()
        rer = make_reread(d0, widths, rr) if res in ("reread", "reread_anchor") else [None] * len(widths)
        prev = d0
        for d, r, q in zip(widths, rows, rer):
            self.layer.append(QuadraticLayer(prev, d, r, batch_norm, net_dropout, shrink, q,
                                             anchor_reread=res == "reread_anchor"))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))
            prev = d
        self.fc = nn.Linear(prev, 1)

    def forward(self, x):
        x0 = x
        for i, layer in enumerate(self.layer):
            x = layer(x, x0)
            if len(self.dropout) > i:
                x = self.dropout[i](x)
        return self.fc(x)


class QNN(BaseModel):
    """QNN-alpha (salmon1802/QNN src/QNN_T26.py, num_heads=1) with per-layer rows; trained with the official
    SE loss: p = (p1 + p2)/2 over two dropout passes, L = BCE(p, y) + BCE(p1, p.detach()) + BCE(p2, p.detach())."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, widths=(624,) * 4, num_rows=(5,) * 4, net_dropout=0.1,
                 batch_norm=False, shrink="slice"):
        super().__init__(vocab_size, num_fields, emb_dim)
        d0 = num_fields * emb_dim
        self.d0, self.widths, self.num_rows = d0, list(widths), list(num_rows)
        self.qnn = QuadraticNetwork(d0, self.widths, self.num_rows, net_dropout, batch_norm, shrink)
        self.reset_parameters()

    def logits(self, feature_emb):
        return self.qnn(feature_emb.flatten(start_dim=1)).squeeze(-1)

    def loss(self, cats, y):
        y = y.float().view(-1)
        x = self.embed(cats).flatten(start_dim=1)
        p1 = torch.sigmoid(self.qnn(x)).view(-1)
        p2 = torch.sigmoid(self.qnn(x)).view(-1)
        p = (p1 + p2) * 0.5
        return (F.binary_cross_entropy(p, y) + F.binary_cross_entropy(p1, p.detach())
                + F.binary_cross_entropy(p2, p.detach()))


class PyramidQNN(QNN):
    """QNN-alpha with narrowing widths (Table 2).  The first layer keeps the width d of the projected input;
    later layers narrow.  With res="reread_anchor" every layer re-reads the input through a parameter-free
    strided slice, as the multiplier and as the skip term."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, d=256, widths=(256, 128, 64, 32), num_rows=(1, 2, 2, 2),
                 net_dropout=0.1, shrink="slice", res="state", rr="slice"):
        BaseModel.__init__(self, vocab_size, num_fields, emb_dim)
        self.d0, self.d, self.widths, self.num_rows = num_fields * emb_dim, d, list(widths), list(num_rows)
        assert d == self.d0  # reads the projected input directly
        # index 0 stays an identity so that parameter names match the released runs
        self.qnn = nn.Sequential(nn.Identity(), QuadraticNetwork(d, self.widths, self.num_rows, net_dropout, False,
                                                                 shrink, res, rr))
        self.reset_parameters()
