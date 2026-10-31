"""First-width-W originals and pyramids for the three added baselines (FCN, GDCN, FiBiNet++), following the
Table-1 rules of pyramid_baselines.py: the original keeps its own design on a W-wide first layer; the pyramid keeps
the same first layer, halves the later layers and never expands (no learned map has out > in).

    FCN (B)       O_FCN: bias-free d0 -> W projection, then the official FCN_sp towers (E2LCN || L2ECN) at width W.
                  P_FCN: same projection; every tower layer maps d_{i-1} -> d_i (halving), residual and anchors
                         shrunk by `shrink` (strided slice or learned down-projection).
    GDCN (B)      O_GDCN: projection, gated cross layers at width W, DNN [W]*3 on the projected input.
                  P_GDCN: projection, gated cross layers [W, W/2, W/4], DNN [W, W/2, W/4].
    FiBiNet++ (A) original = official FiBiNet++ with DNN [W]*3 (SENet+ 'bit' excitation W_2: 52 -> 624 expands).
                  P_FiBiNetPP: SENet+ with field-wise excitation (W_2: 52 -> F, broadcast over the embedding
                         dimension without parameters, as in FiBiNET), DNN [W, W/2, W/4].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pyramid_baselines as pb
from . import x4_baselines as xb
from .extra_baselines import FiBiNetPP, GateCrossLayer, LCACrossNetwork, _tf_glorot_normal_, _tri_bce
from .x4_baselines import MLP_Block, _FuxiCTRModel, multihead_feature_emb


def halve(w, n):
    return [w >> i for i in range(n)]


# --------------------------------------------------------------------------- FCN
class _FCNBase(_FuxiCTRModel):
    def branch_logits(self, cats):
        x = self.proj(multihead_feature_emb(self.embed(cats), 1))
        if getattr(self, "fmask", None) is not None:  # one first-layer mask, computed once, used by both towers
            eta = self.fmask(x)
            return self.tower_a(x, eta).mean(dim=1).squeeze(-1), self.tower_b(x, eta).mean(dim=1).squeeze(-1)
        return self.tower_a(x).mean(dim=1).squeeze(-1), self.tower_b(x).mean(dim=1).squeeze(-1)

    def forward(self, dense, cats):
        a, b = self.branch_logits(cats)
        return (a + b) * 0.5

    def loss(self, cats, y):
        return _tri_bce(*self.branch_logits(cats), y)

    def width_checks(self):
        return []


def _fcn_anchors(exp_num_layers, lin_num_layers):
    return (["self"] * exp_num_layers + ["x0"] * lin_num_layers, ["x0"] * lin_num_layers + ["self"] * exp_num_layers)


class O_FCN(_FCNBase):
    def __init__(self, vocab_size, num_fields=39, emb_dim=16, d=256, exp_num_layers=4, lin_num_layers=2,
                 net_dropout=0.1, batch_norm=False, layer_norm=False):
        super().__init__(vocab_size, num_fields, emb_dim)
        self.d0, self.d = num_fields * emb_dim, d
        self.proj = nn.Linear(self.d0, d, bias=False)
        aa, ab = _fcn_anchors(exp_num_layers, lin_num_layers)
        kw = dict(batch_norm=batch_norm, layer_norm=layer_norm, net_dropout=net_dropout, num_heads=1)
        self.tower_a, self.tower_b = LCACrossNetwork(d, aa, **kw), LCACrossNetwork(d, ab, **kw)
        self.reset_parameters()

    def stacks(self):
        n = len(self.tower_a.anchors)
        return {"tower_a": [self.d] * (n + 1), "tower_b": [self.d] * (n + 1)}


class PyramidLCATower(nn.Module):
    """FCN@KDD'26 LCA cross layer generalized to d_{i-1} -> d_i: H1 = W x (d_{i-1} -> d_i/2), H2 = gamma*H1+beta
    (or LN), H = [H1, H2]; x' = anchor * (H + b) + S(x); anchor = S(x) ('self', ECN) or P(x1) ('x0', LCN)."""

    def __init__(self, d_in, widths, anchors, batch_norm=False, layer_norm=False, net_dropout=0.1, shrink="slice",
                 res="state", rr="proj"):
        super().__init__()
        assert len(widths) == len(anchors)
        self.reread = pb.make_reread(d_in, widths, rr) if res in ("reread", "reread_anchor") else None  # residual r_i(x1)
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
            self.skip.append(pb.make_shrink(prev, d, shrink))
            cum = cum[pb.even_idx(prev, d)] if d != prev else cum
            self.anchor.append(pb.make_shrink(d_in, d, shrink, idx=cum.clone() if (shrink == "slice" and d != d_in)
                                              else None) if a == "x0" else nn.Identity())
            prev = d
        self.fc = nn.Linear(prev, 1)

    def forward(self, x, eta=None):
        x1 = x
        for i, a in enumerate(self.anchors):
            H1 = self.w[i](x)
            H2 = self.layer_norm[i](H1) if len(self.layer_norm) > i else self.gamma[i] * H1 + self.beta[i]
            H = torch.cat([H1, H2], dim=-1)
            if len(self.batch_norm) > i:
                H = self.batch_norm[i](H)
            s = self.reread[i](x1) if self.anchor_reread else self.skip[i](x)
            Hb = H + self.b[i] if (i or eta is None) else eta * (H + self.b[i])  # eta: first-layer instance mask
            x = (s if a == "self" else self.anchor[i](x1)) * Hb + (s if self.reread is None else self.reread[i](x1))
            if len(self.dropout) > i:
                x = self.dropout[i](x)
        return self.fc(x)


class P_FCN(_FCNBase):
    def __init__(self, vocab_size, num_fields=39, emb_dim=16, d=256, widths=None, exp_num_layers=4, lin_num_layers=2,
                 net_dropout=0.1, batch_norm=False, layer_norm=False, shrink="slice", res="state", rr="proj"):
        super().__init__(vocab_size, num_fields, emb_dim)
        n = exp_num_layers + lin_num_layers
        self.d0, self.d, self.widths = num_fields * emb_dim, d, list(widths or halve(d, n))
        # at the native width d = d0 the towers read the embedding directly, as the official FCN does
        self.proj = nn.Identity() if d == self.d0 else nn.Linear(self.d0, d, bias=False)
        aa, ab = _fcn_anchors(exp_num_layers, lin_num_layers)
        kw = dict(batch_norm=batch_norm, layer_norm=layer_norm, net_dropout=net_dropout, shrink=shrink, res=res, rr=rr)
        self.tower_a = PyramidLCATower(d, self.widths, aa, **kw)
        self.tower_b = PyramidLCATower(d, self.widths, ab, **kw)
        self.reset_parameters()

    def stacks(self):
        return {"tower_a": [self.d] + self.widths, "tower_b": [self.d] + self.widths}

    def width_checks(self):
        prevs = [self.d] + self.widths[:-1]
        return [(f"layer{i} concat [H1, H2] <= d_in", w, p) for i, (p, w) in enumerate(zip(prevs, self.widths))]


# --------------------------------------------------------------------------- GDCN
class O_GDCN(_FuxiCTRModel):
    def __init__(self, vocab_size, num_fields=39, emb_dim=16, d=256, num_cross_layers=3, dnn_hidden_units=None,
                 dnn_activations="ReLU", net_dropout=0.5, batch_norm=True):
        super().__init__(vocab_size, num_fields, emb_dim)
        self.d0, self.d = num_fields * emb_dim, d
        self.dnn_widths = list(dnn_hidden_units or [d] * 3)
        self.proj = nn.Linear(self.d0, d, bias=False)
        self.cross_net = GateCrossLayer(d, num_cross_layers)
        self.dnn = MLP_Block(d, self.dnn_widths, dnn_activations, None, None, net_dropout, batch_norm)
        self.fc = nn.Linear(d + self.dnn_widths[-1], 1)
        self.reset_parameters()

    def logits(self, feature_emb):
        x0 = self.proj(feature_emb.flatten(start_dim=1))
        return self.fc(torch.cat([self.cross_net(x0), self.dnn(x0)], dim=1)).squeeze(-1)

    def stacks(self):
        return {"cross": [self.d] * (self.cross_net.cn_layers + 1), "dnn": [self.d] + self.dnn_widths}

    def width_checks(self):
        return []


class PyramidGateCross(nn.Module):
    """GDCN gated cross layer generalized to d_{i-1} -> d_i:
    x_i = P_i(x0) * (W_c x_{i-1} + b) * sigmoid(W_g x_{i-1}) + S_i(x_{i-1}),  W_c, W_g: d_{i-1} -> d_i."""

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
            self.skip.append(pb.make_shrink(prev, d, shrink))
            cum = cum[pb.even_idx(prev, d)] if d != prev else cum
            self.anchor.append(pb.make_shrink(d_in, d, shrink, idx=cum.clone() if (shrink == "slice" and d != d_in)
                                              else None))
            prev = d
        self.out_dim = prev

    def forward(self, x):
        x0 = x
        fmask = getattr(self, "fmask", None)  # first-layer instance mask on W_c x + b (the GDCN gate is kept)
        for i in range(len(self.w)):
            a = self.anchor[i](x0)
            c = self.w[i](x) + self.b[i]
            if i == 0 and fmask is not None:
                c = fmask(x) * c
            x = a * c * torch.sigmoid(self.wg[i](x)) + (self.skip[i](x) if self.res == "state" else a)
        return x


class P_GDCN(_FuxiCTRModel):
    def __init__(self, vocab_size, num_fields=39, emb_dim=16, d=256, cross_widths=None, dnn_hidden_units=None,
                 dnn_activations="ReLU", net_dropout=0.5, batch_norm=True, shrink="slice", res="state", rr="proj",
                 in_proj=True):
        super().__init__(vocab_size, num_fields, emb_dim)
        self.d0, self.d = num_fields * emb_dim, d
        assert in_proj or d == self.d0  # in_proj=False: read the input directly (no learned projection)
        self.cross_widths = list(cross_widths or halve(d, 3))
        self.dnn_widths = list(dnn_hidden_units or halve(d, 3))
        self.proj = nn.Linear(self.d0, d, bias=False) if in_proj else nn.Identity()
        self.cross_net = PyramidGateCross(d, self.cross_widths, shrink, res)
        self.dnn = pb.make_mlp(d, self.dnn_widths, dnn_activations, net_dropout, batch_norm, res, rr)
        self.fc = nn.Linear(self.cross_widths[-1] + self.dnn_widths[-1], 1)
        self.reset_parameters()

    def logits(self, feature_emb):
        x0 = self.proj(feature_emb.flatten(start_dim=1))
        return self.fc(torch.cat([self.cross_net(x0), self.dnn(x0)], dim=1)).squeeze(-1)

    def stacks(self):
        return {"cross": [self.d] + self.cross_widths, "dnn": [self.d] + self.dnn_widths}

    def width_checks(self):
        return []


# --------------------------------------------------------------------------- FiBiNet++
class SENetFieldwise(nn.Module):
    """SENet+ (group mean/max squeeze, skip, shared LN) with field-wise excitation: A = W_2 ReLU(W_1 Z) in R^F,
    broadcast over the embedding dimension (FiBiNET's vector-wise re-weighting), so W_2: r -> F never expands."""

    def __init__(self, num_fields, emb_dim, group_num=2, reduction_ratio=3.0):
        super().__init__()
        self.F, self.D, self.g = num_fields, emb_dim, group_num
        squeeze_dim = 2 * num_fields * group_num
        reduction_size = int(max(1.0, squeeze_dim / reduction_ratio))
        self.W_1 = nn.Linear(squeeze_dim, reduction_size, bias=False)
        self.W_2 = nn.Linear(reduction_size, num_fields, bias=False)
        self.reweight_norm = nn.LayerNorm(emb_dim, eps=1e-6)

    def forward(self, V):
        grp = V.reshape(-1, self.F * self.g, self.D // self.g)
        Z = torch.cat([grp.mean(dim=-1), grp.amax(dim=-1)], dim=-1)
        A = self.W_2(F.relu(self.W_1(Z))).unsqueeze(-1)  # B x F x 1
        return self.reweight_norm(V * A + V)


class P_FiBiNetPP(FiBiNetPP):
    def __init__(self, vocab_size, num_fields=39, emb_dim=16, d=256, dnn_hidden_units=None, res="state", rr="proj", **kw):
        keep_senet = kw.pop("keep_senet", False)  # True: the official bit-wise SENet+ (reads the input; may expand)
        super().__init__(vocab_size, num_fields, emb_dim, dnn_hidden_units=list(dnn_hidden_units or halve(d, 3)), **kw)
        self.d = d
        if res == "reread":  # same Linear/act/dropout layers (official init kept), plus re-read residuals
            old = self.dnn
            self.dnn = pb.ReReadMLP(old.mlp[0].in_features, list(dnn_hidden_units or halve(d, 3)),
                                    kw.get("dnn_activation", "relu"), None, None, kw.get("dnn_dropout", 0.0),
                                    kw.get("dnn_use_bn", False), rr=rr)
            self.dnn.mlp = old.mlp
        if not keep_senet:
            self.senet = SENetFieldwise(num_fields, emb_dim, kw.get("senet_group_num", 2), kw.get("senet_reduction_ratio", 3.0))
            with torch.no_grad():
                for w in (self.senet.W_1.weight, self.senet.W_2.weight):
                    _tf_glorot_normal_(w)

    def stacks(self):
        return {"dnn": [self.dnn.mlp[0].in_features] + [m.out_features for m in self.dnn.mlp if isinstance(m, nn.Linear)]}

    def width_checks(self):
        return []


pb._BUILDERS.update({"o_fcn": O_FCN, "p_fcn": P_FCN, "o_gdcn": O_GDCN, "p_gdcn": P_GDCN, "p_fibinetpp": P_FiBiNetPP})
