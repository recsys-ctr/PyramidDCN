"""Selected FCN_p: three ECN layers parallel to three LCN layers.

LCA operators follow salmon1802/FCN@KDD'26 (8eaa864), with
probability-space Tri-BCE and FuxiCTR initialization. The ECN and LCN
components are required for FCN_p; ECN is not a separate baseline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel
from .layers import even_idx, make_shrink, multihead_feature_emb


class LCACrossNetwork(nn.Module):
    """FCNv2.Exponential2LinearCrossNetwork / Linear2ExponentialCrossNetwork; `anchors[i]` picks layer i's anchor.
    Parameters are created in the official order with the official names (w, b, gamma, beta, layer_norm,
    batch_norm, dropout, fc).  Per layer (official _cross_layer):
        H1 = W_i x  (d -> d/2, no bias);  H2 = LN(H1) if layer_norm else gamma_i * H1 + beta_i  ("LCA")
        H = [H1, H2]  (-> d);  (BatchNorm1d over heads);  x = anchor * (H + b_i) + x;  dropout
    anchor 'self' = current x (ECN, exponential order growth), 'x0' = tower input x1 (LCN, linear growth)."""

    def __init__(self, input_dim, anchors, batch_norm=False, layer_norm=False, net_dropout=0.1, num_heads=1):
        super().__init__()
        self.anchors = list(anchors)
        assert all(a in ("self", "x0") for a in self.anchors)
        self.layer_norm, self.batch_norm, self.dropout = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.w, self.b = nn.ModuleList(), nn.ParameterList()
        self.gamma, self.beta = nn.ParameterList(), nn.ParameterList()
        for i in range(len(self.anchors)):
            self.w.append(nn.Linear(input_dim, input_dim // 2, bias=False))
            if layer_norm:
                self.layer_norm.append(nn.LayerNorm(input_dim // 2))
            else:
                self.gamma.append(nn.Parameter(torch.ones(input_dim // 2)))
                self.beta.append(nn.Parameter(torch.zeros(input_dim // 2)))
            self.b.append(nn.Parameter(torch.empty((input_dim,))))
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(num_heads))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))
            nn.init.uniform_(self.b[i].data)
        self.fc = nn.Linear(input_dim, 1)

    def _cross_layer(self, x, x_anchor, i):
        H_1 = self.w[i](x)
        H_2 = self.layer_norm[i](H_1) if len(self.layer_norm) > i else self.gamma[i] * H_1 + self.beta[i]
        H = torch.cat([H_1, H_2], dim=-1)
        if len(self.batch_norm) > i:
            H = self.batch_norm[i](H)
        x = x_anchor * (H + self.b[i]) + x
        if len(self.dropout) > i:
            x = self.dropout[i](x)
        return x

    def forward(self, x):
        x1 = x
        for i, a in enumerate(self.anchors):
            x = self._cross_layer(x, x if a == "self" else x1, i)
        return self.fc(x)


class PyramidLCACrossNetwork(nn.Module):
    """FCN@KDD'26 LCA cross layer generalized to d_{i-1} -> d_i: H1 = W x (d_{i-1} -> d_i/2), H2 = gamma*H1+beta
    (or LN), H = [H1, H2]; x' = anchor * (H + b) + S(x); anchor = S(x) ('self', ECN) or P(x1) ('x0', LCN)."""

    def __init__(self, d_in, widths, anchors, batch_norm=False, layer_norm=False, net_dropout=0.1, shrink="slice",
                 res="state", rr="slice"):
        super().__init__()
        assert len(widths) == len(anchors)
        assert res == "state" and rr == "slice"
        self.anchors = list(anchors)
        self.w, self.b, self.skip, self.anchor = nn.ModuleList(), nn.ParameterList(), nn.ModuleList(), nn.ModuleList()
        self.gamma, self.beta = nn.ParameterList(), nn.ParameterList()
        self.layer_norm, self.batch_norm, self.dropout = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        prev, cum = d_in, torch.arange(d_in)
        for i, (d, a) in enumerate(zip(widths, anchors)):
            assert d % 2 == 0 and (d <= prev or (i == 0 and shrink == "slice")), (prev, d)  # first layer may copy-widen
            self.w.append(nn.Linear(prev, d // 2, bias=False))
            if layer_norm:
                self.layer_norm.append(nn.LayerNorm(d // 2))
            else:
                self.gamma.append(nn.Parameter(torch.ones(d // 2)))
                self.beta.append(nn.Parameter(torch.zeros(d // 2)))
            self.b.append(nn.Parameter(torch.empty((d,))))
            nn.init.uniform_(self.b[i].data)
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(1))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))
            self.skip.append(make_shrink(prev, d, shrink))
            cum = cum[even_idx(prev, d)] if d != prev else cum
            self.anchor.append(make_shrink(d_in, d, shrink, idx=cum.clone() if (shrink == "slice" and d != d_in)
                                              else None) if a == "x0" else nn.Identity())
            prev = d
        self.fc = nn.Linear(prev, 1)

    def forward(self, x):
        x1 = x
        for i, a in enumerate(self.anchors):
            H1 = self.w[i](x)
            H2 = self.layer_norm[i](H1) if len(self.layer_norm) > i else self.gamma[i] * H1 + self.beta[i]
            H = torch.cat([H1, H2], dim=-1)
            if len(self.batch_norm) > i:
                H = self.batch_norm[i](H)
            s = self.skip[i](x)
            Hb = H + self.b[i]
            x = (s if a == "self" else self.anchor[i](x1)) * Hb + s
            if len(self.dropout) > i:
                x = self.dropout[i](x)
        return self.fc(x)


class FCNP(BaseModel):
    """Three-layer ECN and LCN in parallel, with probability-space Tri-BCE."""
    def __init__(self, vocab_size, num_fields, emb_dim, widths, converted=False):
        super().__init__(vocab_size, num_fields, emb_dim)
        assert len(widths) == 3
        d = num_fields * emb_dim
        kwargs = dict(net_dropout=0., batch_norm=False, layer_norm=False)
        if converted:
            self.ECN = PyramidLCACrossNetwork(d, widths, ["self"] * 3, **kwargs)
            self.LCN = PyramidLCACrossNetwork(d, widths, ["x0"] * 3, **kwargs)
        else:
            assert widths == [d] * 3
            self.ECN = LCACrossNetwork(d, ["self"] * 3, **kwargs)
            self.LCN = LCACrossNetwork(d, ["x0"] * 3, **kwargs)
        self.reset_parameters()
        if converted:
            self.LCN.anchor = nn.ModuleList(StridedSlice(idx) for idx in balanced_idx(d, widths))

    def branch_logits(self, cats):
        x = multihead_feature_emb(self.embed(cats), 1)
        return self.ECN(x).mean(dim=1).squeeze(-1), self.LCN(x).mean(dim=1).squeeze(-1)

    def forward(self, dense, cats):
        a, b = self.branch_logits(cats)
        return (a + b) * 0.5

    def loss(self, cats, y):
        return original_tri_bce(*self.branch_logits(cats), y)

from .layers import StridedSlice, balanced_idx

def original_tri_bce(a, b, labels):
    y = labels.float().reshape(-1)
    p, pa, pb = torch.sigmoid((a + b) * 0.5), torch.sigmoid(a), torch.sigmoid(b)
    main = F.binary_cross_entropy(p, y)
    la = F.binary_cross_entropy(pa, y)
    lb = F.binary_cross_entropy(pb, y)
    wa = torch.where(la > main, la - main, torch.zeros_like(la))
    wb = torch.where(lb > main, lb - main, torch.zeros_like(lb))
    loss = main + la * wa + lb * wb
    return loss
