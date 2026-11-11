"""FCN and its pyramidal conversion (Table 2).

    FCN          hand port of FCN_sp = salmon1802/FCN@KDD'26 (8eaa864) src/FCNv2.py:
                 Exponential2LinearCrossNetwork || Linear2ExponentialCrossNetwork, LCA cross layers, Tri-BCE
    PyramidFCN   the two towers with per-layer widths

Initialisation: FuxiCTR conventions (embedding N(0, 1e-4^2); exact-type nn.Linear -> xavier_normal_, bias 0;
custom params keep their own init: b ~ U(0,1), gamma=1, beta=0).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel
from .layers import even_idx, make_reread, make_shrink, multihead_feature_emb


def tri_bce(logit_a, logit_b, y):
    """FCNv1/FCNv2.add_loss verbatim (BCE on sigmoid == BCE-with-logits), weights not detached:
    L = L_y + L_a * max(L_a - L_y, 0) + L_b * max(L_b - L_y, 0)."""
    y = y.float().view(-1)
    loss = F.binary_cross_entropy_with_logits((logit_a + logit_b) * 0.5, y)
    loss_a = F.binary_cross_entropy_with_logits(logit_a, y)
    loss_b = F.binary_cross_entropy_with_logits(logit_b, y)
    w_a, w_b = loss_a - loss, loss_b - loss
    w_a = torch.where(w_a > 0, w_a, torch.zeros_like(w_a))
    w_b = torch.where(w_b > 0, w_b, torch.zeros_like(w_b))
    return loss + loss_a * w_a + loss_b * w_b


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


class FCN(BaseModel):
    """salmon1802/FCN@KDD'26 src/FCNv2.py (FCN_sp: E2LCN || L2ECN).  Multi-head embedding (table width
    emb_dim*num_heads, per-head input F*emb_dim); served logit = (mean_h tower_a + mean_h tower_b) / 2; trained
    with Tri-BCE."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, structure="sp", exp_num_layers=4, lin_num_layers=2,
                 net_dropout=0.1, batch_norm=False, layer_norm=False, num_heads=1):
        super().__init__(vocab_size, num_fields, emb_dim, table_dim=emb_dim * num_heads)
        assert structure == "sp", structure
        self.num_heads = num_heads
        input_dim = num_fields * emb_dim  # feature_map.sum_emb_out_dim()
        kw = dict(net_dropout=net_dropout, batch_norm=batch_norm, layer_norm=layer_norm, num_heads=num_heads)
        E, L = exp_num_layers, lin_num_layers
        self.E2LCN = LCACrossNetwork(input_dim, ["self"] * E + ["x0"] * L, **kw)  # official tower names
        self.L2ECN = LCACrossNetwork(input_dim, ["x0"] * L + ["self"] * E, **kw)
        self.reset_parameters()

    def branch_logits(self, cats):
        x = multihead_feature_emb(self.embed(cats), self.num_heads)  # B x H x F*D
        return self.E2LCN(x).mean(dim=1).squeeze(-1), self.L2ECN(x).mean(dim=1).squeeze(-1)

    def forward(self, dense, cats):
        la, lb = self.branch_logits(cats)
        return (la + lb) * 0.5

    def loss(self, cats, y):
        return tri_bce(*self.branch_logits(cats), y)


def _fcn_anchors(exp_num_layers, lin_num_layers):
    return (["self"] * exp_num_layers + ["x0"] * lin_num_layers, ["x0"] * lin_num_layers + ["self"] * exp_num_layers)


class PyramidLCACrossNetwork(nn.Module):
    """FCN@KDD'26 LCA cross layer generalized to d_{i-1} -> d_i: H1 = W x (d_{i-1} -> d_i/2), H2 = gamma*H1+beta
    (or LN), H = [H1, H2]; x' = anchor * (H + b) + S(x); anchor = S(x) ('self', ECN) or P(x1) ('x0', LCN)."""

    def __init__(self, d_in, widths, anchors, batch_norm=False, layer_norm=False, net_dropout=0.1, shrink="slice",
                 res="state", rr="slice"):
        super().__init__()
        assert len(widths) == len(anchors)
        self.reread = make_reread(d_in, widths, rr) if res in ("reread", "reread_anchor") else None  # residual r_i(x1)
        self.anchor_reread = res == "reread_anchor"  # 'self' layers also multiply by r_i(x1) instead of S(x)
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
            s = self.reread[i](x1) if self.anchor_reread else self.skip[i](x)
            Hb = H + self.b[i]
            x = (s if a == "self" else self.anchor[i](x1)) * Hb + (s if self.reread is None else self.reread[i](x1))
            if len(self.dropout) > i:
                x = self.dropout[i](x)
        return self.fc(x)


class PyramidFCN(BaseModel):
    """FCN_sp towers (E2LCN || L2ECN, Tri-BCE) whose layers map d_{i-1} -> d_i (Table 2).  The residual and
    the x0 anchors are cut by strided slices, or every layer re-reads the input (res="reread")."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, d=256, widths=None, exp_num_layers=4, lin_num_layers=2,
                 net_dropout=0.1, batch_norm=False, layer_norm=False, shrink="slice", res="state", rr="slice"):
        super().__init__(vocab_size, num_fields, emb_dim)
        self.d0, self.d, self.widths = num_fields * emb_dim, d, list(widths)
        assert d == self.d0, "the towers read the projected input directly"
        aa, ab = _fcn_anchors(exp_num_layers, lin_num_layers)
        kw = dict(batch_norm=batch_norm, layer_norm=layer_norm, net_dropout=net_dropout, shrink=shrink, res=res, rr=rr)
        self.tower_a = PyramidLCACrossNetwork(d, self.widths, aa, **kw)
        self.tower_b = PyramidLCACrossNetwork(d, self.widths, ab, **kw)
        self.reset_parameters()

    def branch_logits(self, cats):
        x = multihead_feature_emb(self.embed(cats), 1)
        return self.tower_a(x).mean(dim=1).squeeze(-1), self.tower_b(x).mean(dim=1).squeeze(-1)

    def forward(self, dense, cats):
        a, b = self.branch_logits(cats)
        return (a + b) * 0.5

    def loss(self, cats, y):
        return tri_bce(*self.branch_logits(cats), y)
