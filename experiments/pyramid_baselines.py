"""Pyramidized, no-up-projection versions of the x4_baselines.py ports (Criteo_x4_001, d0 = 39*16 = 624).

API (same as x4_baselines.build_baseline):
    build_pyramid(name, vocab_size, num_fields=39, emb_dim=16, widths="same", **kw) -> nn.Module
      .embedding (shared nn.Embedding), .forward(dense, cats) -> logits [B], .loss(cats, y)
    names   : p_dcnv2, p_finalmlp, p_masknet, p_qnn, p_ecn
    widths  : "same" | "iso" | "official" (non-compliant reference = the official model) | dict of width kwargs
    kw      : variant switches, e.g. shrink="slice"|"proj" (p_dcnv2, p_ecn, p_qnn), mask_space="input"|"output"
              (p_masknet); any other constructor kwarg.

Rules enforced (and checked by `audit`):
  R1  every learned Linear / einsum weight maps out <= in, except scalar output heads (out_features == 1, the
      FinalMLP bilinear scalar fusion).  A fused Linear producing R rows is one Linear(d_in, R*d_out), so it is
      covered by the same check.  Parallel branches that read the same input and are concatenated into a
      representation consumed by a further learned non-scalar layer (MaskNet's parallel blocks -> DNN) count
      as fused rows: sum of branch widths <= input width.  Branches that only meet in the scalar head
      (DCNv2 cross || DNN -> fc, FinalMLP mlp1 / mlp2 -> fusion) are not fused rows.
  R2  every stack's widths are non-increasing with depth.
Everything else (operators, norms, BN, dropout, activations, init, loss incl. Tri-BCE / SE loss) follows
x4_baselines.py / the official Criteo configs.

Width changes on residual / anchor paths ("shrink"):
  "slice": parameter-free evenly-spaced (strided) coordinate selection idx_j = floor(j * d_in / d_out)
           (== x[..., ::2] for halving).  A strided rather than prefix slice keeps every field represented: the
           flattened embedding is field-major, so a prefix slice 624 -> 156 would drop 29.5 of the 39 fields
           from the skip/anchor path.  For DCNv2's x0 anchor the indices are composed through the layers, so
           P_i x0 and R_i x_{i-1} always select the same original coordinates.
  "proj" : learned bias-free Linear(d_in, d_out) (out < in, R1-compliant), FuxiCTR xavier init.
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import x4_baselines as xb  # noqa: E402
from .x4_baselines import MLP_Block, MaskBlock, _FuxiCTRModel, get_activation, multihead_feature_emb  # noqa: E402

serving_macs, dense_params = xb.serving_macs, xb.dense_params


# --------------------------------------------------------------------------- width-change helpers
class StridedSlice(nn.Module):
    """Parameter-free coordinate selection (non-persistent index buffer, so state_dicts stay official-compatible)."""

    def __init__(self, idx):
        super().__init__()
        self.register_buffer("idx", torch.as_tensor(idx, dtype=torch.long), persistent=False)

    def forward(self, x):
        return x.index_select(-1, self.idx)


def even_idx(d_in, d_out):
    return (torch.arange(d_out) * d_in) // d_out


def balanced_idx(d0, dims):
    """Least-used re-read: layer i keeps the d_i input coordinates that
    layers 1..i-1 used least, evenly spaced among equally used ones (a layer wider than the input first takes every
    coordinate once, then continues with the least used).  Overlap between layers is thus the minimum possible and usage
    counts never differ by more than one.  For halving widths it gives 1,3,5,... | 2,6,10,... | 4,12,20,... (1-based)."""
    count, out = [0] * d0, []
    for d in dims:
        pick, need = [], d
        while need > 0:
            m = min(count)
            tied = [c for c in range(d0) if count[c] == m]
            take = tied if need >= len(tied) else [tied[int(v)] for v in even_idx(len(tied), need)]
            for c in take:
                count[c] += 1
            pick += take
            need -= len(take)
        out.append(sorted(pick))
    return out


def field_split_idx(field_dims, dims, alpha=1.0):
    """Field-aware least-used re-read: layer i gives field f a quota proportional to E_f^alpha (alpha = 1: proportional
    to its size, alpha = 0: equal per field) with at least one coordinate per field (largest-remainder rounding, so the
    quotas sum to d_i); inside the field it takes its least-used coordinates, evenly spaced among ties.  Every narrowed
    layer sees every field; coordinates are reused only where a field has fewer coordinates than its quotas need."""
    starts = [sum(field_dims[:f]) for f in range(len(field_dims))]
    d0, F = sum(field_dims), len(field_dims)
    count, out = [0] * d0, []
    for d in dims:
        if d >= d0:
            pick = [int(v) for v in even_idx(d0, d)] if d > d0 else list(range(d0))
            for c in pick: count[c] += 1
            out.append(sorted(pick)); continue
        assert d >= F, "a layer narrower than the number of fields cannot see every field"
        q, cap = [1] * F, list(field_dims)                                # one per field, the rest by weight
        while sum(q) < d:                                                 # (capped at the field size)
            room = [f for f in range(F) if q[f] < cap[f]]
            w = {f: field_dims[f] ** alpha for f in room}
            left = d - sum(q)
            raw = {f: left * w[f] / sum(w.values()) for f in room}
            add = {f: min(int(raw[f]), cap[f] - q[f]) for f in room}
            if sum(add.values()) == 0:                                    # hand out the remainder one by one
                for f in sorted(room, key=lambda f: raw[f], reverse=True)[:left]:
                    add[f] = 1
            for f in room:
                q[f] += add[f]
        pick = []
        for s, n, k in zip(starts, field_dims, q):
            cs = list(range(s, s + n)); taken = []
            while len(taken) < k:                                         # least-used inside the field
                m = min(count[c] for c in cs if c not in taken)
                tied = [c for c in cs if count[c] == m and c not in taken]
                need = k - len(taken)
                take = tied if need >= len(tied) else [tied[int(v)] for v in even_idx(len(tied), need)]
                taken += take
                for c in take: count[c] += 1
            pick += taken
        out.append(sorted(pick))
    return out


def field_plan_idx(field_dims, dims):
    """Planned field split for fields of different sizes: the widths of the
    narrowed layers are known, so each field f is split across them in advance, layer i getting about E_f * d_i / d0
    of its coordinates (wider layers first, largest-remainder rounding, never more than the field has left), and the
    split is disjoint: a one-dimensional field enters one layer.  Inside a field each layer takes its coordinates evenly
    spaced among those still free.  Layers at least as wide as x read all of it.  If the narrowed widths exceed d0
    (overlap unavoidable) it falls back to field_split_idx."""
    F, d0 = len(field_dims), sum(field_dims)
    starts = [sum(field_dims[:f]) for f in range(F)]
    narrow = [i for i, d in enumerate(dims) if d < d0]
    if sum(dims[i] for i in narrow) > d0:
        return field_split_idx(field_dims, dims)
    cap, quota = list(field_dims), {i: [0] * F for i in narrow}
    for i in sorted(narrow, key=lambda i: -dims[i]):               # plan the quotas, widest layer first
        target = [n * dims[i] / d0 for n in field_dims]
        q = [min(int(t), c) for t, c in zip(target, cap)]
        while sum(q) < dims[i]:
            room = sorted((f for f in range(F) if q[f] < cap[f]), key=lambda f: target[f] - q[f], reverse=True)
            for f in room[: dims[i] - sum(q)]:
                q[f] += 1
        quota[i] = q
        cap = [c - k for c, k in zip(cap, q)]
    free = [list(range(s, s + n)) for s, n in zip(starts, field_dims)]
    out = []
    for i, d in enumerate(dims):
        if d >= d0:
            out.append([int(v) for v in even_idx(d0, d)] if d > d0 else list(range(d0))); continue
        pick = []
        for f in range(F):                                          # inside the field: evenly spaced among free ones
            k = quota[i][f]
            if k:
                take = [free[f][int(v)] for v in even_idx(len(free[f]), k)]
                free[f] = [c for c in free[f] if c not in set(take)]
                pick += take
        out.append(sorted(pick))
    return out


def disjoint_idx(d0, dims):
    """General disjoint re-read: layer i keeps d_i input coordinates evenly spaced among those not kept
    by layers 1..i-1; a layer at least as wide as the input keeps (copies) all of them; when fewer than d_i coordinates
    remain, the pool restarts from all d0.  For halving widths [W, W/2, W/4, ...] this gives 1,3,5,... | 2,6,10,... |
    4,12,20,... (1-based), disjoint at any depth; for three layers it equals the offset rule of anchor="stagger"."""
    used, out = set(), []
    for d in dims:
        if d >= d0:
            out.append(list(range(d0)) if d == d0 else [int(v) for v in even_idx(d0, d)])
            used = set(range(d0))
            continue
        pool = [c for c in range(d0) if c not in used]
        if len(pool) < d:
            used, pool = set(), list(range(d0))
        pick = [pool[int(v)] for v in even_idx(len(pool), d)]
        used |= set(pick)
        out.append(pick)
    return out


def make_shrink(d_in, d_out, mode, idx=None):
    if d_out == d_in and idx is None:
        return nn.Identity()
    assert d_out <= d_in or mode == "slice", (d_in, d_out)  # slice may widen the input by copying coordinates
    if mode == "slice":
        return StridedSlice(even_idx(d_in, d_out) if idx is None else idx)
    if mode == "proj":
        return nn.Linear(d_in, d_out, bias=False)
    raise ValueError(mode)


def _nonincreasing(ws):
    return all(b <= a for a, b in zip(ws, ws[1:]))


def make_reread(d_src, widths, rr):
    """Re-read residuals r_i: d_src -> d_i (bias-free learned map for rr="proj", strided slice for rr="slice";
    Identity when d_i == d_src).  Learned maps get the FuxiCTR init (xavier_normal_)."""
    mods = nn.ModuleList(make_shrink(d_src, d, rr) for d in widths)
    for m in mods:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
    return mods


class ReReadMLP(MLP_Block):
    """MLP_Block whose hidden layer i adds a re-read of the block input: h_i = block_i(h_{i-1}) + r_i(x)
    (the residual of the narrowing design applied to an MLP; widths must not exceed the input width).
    Same `mlp` Sequential as MLP_Block, so official init code that walks `mlp` still applies."""

    def __init__(self, input_dim, hidden_units=(), hidden_activations="ReLU", output_dim=None, output_activation=None,
                 dropout_rates=0.0, batch_norm=False, use_bias=True, rr="proj"):
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


def make_mlp(input_dim, hidden_units, act, dropout, batch_norm, res="state", rr="proj"):
    if res == "reread":
        return ReReadMLP(input_dim, hidden_units, act, None, None, dropout, batch_norm, rr=rr)
    return MLP_Block(input_dim, hidden_units, act, None, None, dropout, batch_norm)


# --------------------------------------------------------------------------- p_dcnv2
class PyramidCrossNet(nn.Module):
    """x_i = P_i x0 * (W_i x_{i-1} + b_i) + R_i x_{i-1};  W_i: d_{i-1} -> d_i (bias), P_i: d0 -> d_i,
    R_i: d_{i-1} -> d_i (Identity when widths are equal).  widths == [d0]*L is exactly CrossNetV2."""

    def __init__(self, d0, widths, shrink="proj", res="state"):
        super().__init__()
        self.res = res  # "reread": the residual is the anchor P_i X_0 (a re-read of the input), not R_i X_{i-1}
        self.cross_layers, self.anchor, self.skip = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        prev, cum = d0, torch.arange(d0)
        for i, d in enumerate(widths):
            assert d <= prev or (i == 0 and shrink == "slice")  # only the first layer may widen (input-side copy)
            self.cross_layers.append(nn.Linear(prev, d))
            rel = even_idx(prev, d)
            self.skip.append(make_shrink(prev, d, shrink) if res == "state" else nn.Identity())
            cum = cum[rel] if d != prev else cum
            self.anchor.append(make_shrink(d0, d, shrink, idx=cum.clone() if (shrink == "slice" and d != d0) else None))
            prev = d

    def forward(self, X_0):
        X_i = X_0
        fmask = getattr(self, "fmask", None)  # first-layer instance mask (attach_first_mask), multiplies W_1 x + b_1
        for i, (W, P, R) in enumerate(zip(self.cross_layers, self.anchor, self.skip)):
            A = P(X_0)
            c = W(X_i) if (i or fmask is None) else fmask(X_i) * W(X_i)
            X_i = (R(X_i) if self.res == "state" else A) + A * c  # official order: X_i + X_0 * W(X_i)
        return X_i


class PDCNv2(_FuxiCTRModel):
    """DCNv2 (parallel) with a narrowing cross net and a non-increasing parallel DNN (BN, ReLU, dropout 0.1)."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, cross_widths=(624,) * 3, dnn_widths=(624,) * 5,
                 shrink="proj", dnn_activations="relu", net_dropout=0.1, batch_norm=True, res="state", rr="proj"):
        super().__init__(vocab_size, num_fields, emb_dim)
        d0 = num_fields * emb_dim
        self.d0, self.cross_widths, self.dnn_widths = d0, list(cross_widths), list(dnn_widths)
        self.crossnet = PyramidCrossNet(d0, self.cross_widths, shrink, res)
        self.parallel_dnn = make_mlp(d0, self.dnn_widths, dnn_activations, net_dropout, batch_norm, res, rr)
        self.fc = nn.Linear(self.cross_widths[-1] + self.dnn_widths[-1], 1)
        self.reset_parameters()

    def logits(self, feature_emb):
        x0 = feature_emb.flatten(start_dim=1)
        return self.fc(torch.cat([self.crossnet(x0), self.parallel_dnn(x0)], dim=-1)).squeeze(-1)

    def stacks(self):
        return {"cross": [self.d0] + self.cross_widths, "dnn": [self.d0] + self.dnn_widths}

    def width_checks(self):
        return []


# --------------------------------------------------------------------------- p_finalmlp
class NarrowingInteractionAggregation(xb.InteractionAggregation):
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

    def einsum_maps(self):
        dx, dy = self.head_x_dim, self.head_y_dim
        return [("fusion_module.w_xy[per-head contraction]", max(dx, dy), min(dx, dy))]


def fusion_heads(dx, dy, official=50):
    """Largest divisor of gcd(dx, dy) not above the official head count (50)."""
    g = math.gcd(dx, dy)
    return max(h for h in range(1, min(g, official) + 1) if g % h == 0)


class PFinalMLP(_FuxiCTRModel):
    """FinalMLP (use_fs=False, as the official Criteo_x4 config) with pyramid stream widths."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, mlp1_widths=(624,) * 5, mlp2_widths=(500,),
                 num_heads=None, mlp1_dropout=0.2, mlp2_dropout=0.1, activations="relu", batch_norm=False, res="state",
                 rr="proj"):
        super().__init__(vocab_size, num_fields, emb_dim)
        d0 = num_fields * emb_dim
        self.d0, self.mlp1_widths, self.mlp2_widths = d0, list(mlp1_widths), list(mlp2_widths)
        self.num_heads = num_heads or fusion_heads(self.mlp1_widths[-1], self.mlp2_widths[-1])
        self.mlp1 = make_mlp(d0, self.mlp1_widths, activations, mlp1_dropout, batch_norm, res, rr)
        self.mlp2 = make_mlp(d0, self.mlp2_widths, activations, mlp2_dropout, batch_norm, res, rr)
        self.fusion_module = NarrowingInteractionAggregation(self.mlp1_widths[-1], self.mlp2_widths[-1], 1,
                                                             self.num_heads)
        self.reset_parameters()

    def logits(self, feature_emb):
        flat = feature_emb.flatten(start_dim=1)
        return self.fusion_module(self.mlp1(flat), self.mlp2(flat)).squeeze(-1)

    def stacks(self):
        return {"mlp1": [self.d0] + self.mlp1_widths, "mlp2": [self.d0] + self.mlp2_widths}

    def width_checks(self):
        return []


# --------------------------------------------------------------------------- p_masknet
class OutputMaskBlock(nn.Module):
    """Output-space MaskBlock: act(LN?(mask(V_emb) * W_h V_hidden)) -> dropout, with
    mask = W2 ReLU(W1 V_emb), W1: d_in -> k, W2: k -> d_block, d_block <= k <= d_in (no up-projection)."""

    def __init__(self, input_dim, hidden_dim, output_dim, mask_hidden, hidden_activation="ReLU", dropout_rate=0,
                 layer_norm=True):
        super().__init__()
        assert output_dim <= mask_hidden <= input_dim
        self.mask_layer = nn.Sequential(nn.Linear(input_dim, mask_hidden), nn.ReLU(),
                                        nn.Linear(mask_hidden, output_dim))
        self.proj = nn.Linear(hidden_dim, output_dim, bias=False)
        post = [nn.LayerNorm(output_dim)] if layer_norm else []
        post.append(get_activation(hidden_activation))
        if dropout_rate > 0:
            post.append(nn.Dropout(p=dropout_rate))
        self.hidden_layer = nn.Sequential(*post)

    def forward(self, V_emb, V_hidden):
        return self.hidden_layer(self.mask_layer(V_emb) * self.proj(V_hidden))


class PParallelMaskNet(nn.Module):
    def __init__(self, blocks, dnn_input, dnn_widths, activations, dropout):
        super().__init__()
        self.mask_blocks = nn.ModuleList(blocks)
        self.dnn = MLP_Block(dnn_input, dnn_widths, activations, 1, None, dropout)  # official Sigmoid -> loss

    def forward(self, V_emb, V_hidden):
        return self.dnn(torch.cat([blk(V_emb, V_hidden) for blk in self.mask_blocks], dim=-1))


class PMaskNet(_FuxiCTRModel):
    """ParallelMaskNet (2 blocks, dropout 0.3, no LN) with a compliant mask and a non-increasing DNN.
    mask_space="input": the official MaskBlock unchanged with reduction_ratio = 1 (mask MLP d0 -> d0 -> d0: the
                        only ratio for which an input-space mask has no up-projection; it is FuxiCTR's default).
    mask_space="output": OutputMaskBlock, mask computed in the block's output space (d0 -> k -> d_block)."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, block_dim=312, dnn_widths=(624,) * 5, num_blocks=2,
                 mask_space="input", reduction_ratio=1.0, mask_hidden=None, net_dropout=0.3, activations="relu",
                 net_layernorm=False):
        super().__init__(vocab_size, num_fields, emb_dim)
        d0 = num_fields * emb_dim
        self.d0, self.block_dim, self.num_blocks, self.dnn_widths = d0, block_dim, num_blocks, list(dnn_widths)
        self.mask_space, self.reduction_ratio = mask_space, reduction_ratio
        self.mask_hidden = mask_hidden or block_dim
        if mask_space == "input":
            blocks = [MaskBlock(d0, d0, block_dim, activations, reduction_ratio, net_dropout, net_layernorm)
                      for _ in range(num_blocks)]
        elif mask_space == "output":
            blocks = [OutputMaskBlock(d0, d0, block_dim, self.mask_hidden, activations, net_dropout, net_layernorm)
                      for _ in range(num_blocks)]
        else:
            raise ValueError(mask_space)
        self.mask_net = PParallelMaskNet(blocks, num_blocks * block_dim, self.dnn_widths, activations, net_dropout)
        self.reset_parameters()

    def logits(self, feature_emb):
        V = feature_emb.flatten(start_dim=1)  # emb_layernorm=False (official): V_hidden = V_emb
        return self.mask_net(V, V).squeeze(-1)

    def stacks(self):
        if self.mask_space == "input":
            mask = [self.d0, int(self.d0 * self.reduction_ratio), self.d0]
        else:
            mask = [self.d0, self.mask_hidden, self.block_dim]
        return {"mask_mlp": mask, "block": [self.d0, self.block_dim],
                "dnn": [self.num_blocks * self.block_dim] + self.dnn_widths}

    def width_checks(self):
        return [("parallel blocks concat (fused rows) <= block input", self.num_blocks * self.block_dim, self.d0)]


# --------------------------------------------------------------------------- p_qnn
class PyramidQuadraticLayer(nn.Module):
    """QNN_T26 QuadraticLayer (num_heads=1) generalized to d_in -> d_out with R rows (R*d_out <= d_in):
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
        if getattr(self, "fmask", None) is not None:  # first-layer instance mask, shared by the R rows
            h = h * self.fmask(x).unsqueeze(1)
        if self.net_dropout > 0:
            h = self.dropout(h)
        return (ego_x.unsqueeze(1) * h).sum(dim=1) + (ego_x if self.reread is None else self.reread(x0))


class PQNNNet(nn.Module):
    def __init__(self, d0, widths, rows, net_dropout, batch_norm, shrink, res="state", rr="proj"):
        super().__init__()
        self.layer, self.dropout = nn.ModuleList(), nn.ModuleList()
        rer = make_reread(d0, widths, rr) if res in ("reread", "reread_anchor") else [None] * len(widths)
        prev = d0
        for d, r, q in zip(widths, rows, rer):
            self.layer.append(PyramidQuadraticLayer(prev, d, r, batch_norm, net_dropout, shrink, q,
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


class PQNN(_FuxiCTRModel):
    """QNN-alpha (salmon1802/QNN src/QNN_T26.py, num_heads=1) with per-layer rows; trained with the official
    SE loss: p = (p1 + p2)/2 over two dropout passes, L = BCE(p, y) + BCE(p1, p.detach()) + BCE(p2, p.detach())."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, widths=(624,) * 4, num_rows=None, net_dropout=0.1,
                 batch_norm=False, shrink="slice", max_rows=5):
        super().__init__(vocab_size, num_fields, emb_dim)
        d0 = num_fields * emb_dim
        self.d0, self.widths = d0, list(widths)
        if num_rows is None:  # as many rows as R1 allows, capped at the official R = 5
            prevs = [d0] + self.widths[:-1]
            num_rows = [max(1, min(max_rows, p // d)) for p, d in zip(prevs, self.widths)]
        elif isinstance(num_rows, int):
            num_rows = [num_rows] * len(self.widths)
        self.num_rows = list(num_rows)
        self.qnn = PQNNNet(d0, self.widths, self.num_rows, net_dropout, batch_norm, shrink)
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

    def stacks(self):
        return {"qnn": [self.d0] + self.widths}

    def width_checks(self):
        prevs = [self.d0] + self.widths[:-1]
        return [(f"layer{i} rows*d_out <= d_in (R={r})", r * d, p)
                for i, (p, d, r) in enumerate(zip(prevs, self.widths, self.num_rows))]


# --------------------------------------------------------------------------- p_ecn
class PyramidECNNet(nn.Module):
    """FCN@master ECN layer generalized to d_in -> d_out (even): H = W x (d_in -> d_out/2, no bias);
    z = [H, H * ReLU(LN(H))] (d_out); x' = S(x) * (z + b) + S(x); dropout.  widths == [d0]*L is the official ECN."""

    def __init__(self, d0, widths, layer_norm=True, batch_norm=False, net_dropout=0.1, num_heads=1, shrink="slice",
                 res="state", rr="proj"):
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
            if i == 0 and getattr(self, "fmask", None) is not None:  # first-layer instance mask on z + b
                zb = self.fmask(x) * zb
            x = s * zb + (s if self.reread is None else self.reread[i](x_in))
            if len(self.dropout) > i:
                x = self.dropout[i](x)
        return self.fc(x)


class PECN(_FuxiCTRModel):
    """ECN (self-mask, official Criteo config: dropout 0.1, LN True, BN False, 1 head, plain BCE)."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, widths=(624,) * 4, net_dropout=0.1, layer_norm=True,
                 batch_norm=False, shrink="slice"):
        super().__init__(vocab_size, num_fields, emb_dim)
        d0 = num_fields * emb_dim
        self.d0, self.widths = d0, list(widths)
        self.ECN = PyramidECNNet(d0, self.widths, layer_norm, batch_norm, net_dropout, 1, shrink)
        self.reset_parameters()

    def logits(self, feature_emb):
        return self.ECN(multihead_feature_emb(feature_emb, 1)).mean(dim=1).squeeze(-1)

    def stacks(self):
        return {"ecn": [self.d0] + self.widths}

    def width_checks(self):
        prevs = [self.d0] + self.widths[:-1]
        return [(f"layer{i} concat [H, H*mask] <= d_in", d, p) for i, (p, d) in enumerate(zip(prevs, self.widths))]


# --------------------------------------------------------------------------- first-layer-256 originals
# Fair-comparison controls: every original starts with the same first layer (a bias-free 624 -> d down-projection,
# d = 256) and then runs its paper's architecture on the d-wide stream, expanding by the ratio its paper states:
#   DCNv2  low-rank cross, rank = input size / 4 (DCN V2 paper, Sec. 8)      -> U: d/4 -> d   (4x)
#   MaskNet instance mask with reduction ratio 2 (MaskNet paper, Sec. 4.1.4) -> W1: d -> 2d   (2x)
#   QNN-a  M = 5 rows (official QNN_T26 config; peak of the paper's Fig. 7) -> W: d -> 5d    (5x)
#   ML-DCN low-rank r = d/4 (no ratio in the paper; DCN V2 rule), mask ratio t = 0.5 (ML-DCN paper)
# These models are intentionally NOT R1-compliant (audit lists their expansions).
class O256DCNv2(_FuxiCTRModel):
    def __init__(self, vocab_size, num_fields=39, emb_dim=16, d=256, num_cross_layers=3, rank=None,
                 dnn_widths=(256,) * 5, net_dropout=0.1, batch_norm=True, full_rank=False, project=True,
                 up_frozen=False, cross_widths=None, ranks=None):
        super().__init__(vocab_size, num_fields, emb_dim)
        d0 = num_fields * emb_dim
        if not project:  # stream stays at the input width d0; the first map V: d0 -> rank is the first layer
            d = d0
        r = d if full_rank else rank or d // 4
        self.d0, self.d, self.r, self.dnn_widths = d0, d, r, list(dnn_widths)
        self.cross_widths = list(cross_widths) if cross_widths is not None else None
        self.proj = nn.Linear(d0, d, bias=False) if project else nn.Identity()
        if full_rank:  # CrossNetV2: x + x0 * (W x + b), W: d -> d (constant width, no expansion)
            self.V = nn.ModuleList(nn.Identity() for _ in range(num_cross_layers))
            self.U = nn.ModuleList(nn.Linear(d, d) for _ in range(num_cross_layers))
        elif self.cross_widths is None:
            self.V = nn.ModuleList(nn.Linear(d, r, bias=False) for _ in range(num_cross_layers))
            self.U = nn.ModuleList(nn.Linear(r, d) for _ in range(num_cross_layers))
        else:  # no restoring map: layer l writes U: r_l -> cross_widths[l]; state and x0 anchor cut by strided slices
            # ranks: per-layer rank r_l (default: r everywhere); the first layer may widen by copying input coordinates
            assert len(self.cross_widths) == num_cross_layers
            rk = list(ranks) if ranks is not None else [r] * num_cross_layers
            self.V, self.U, self.skip, self.anchor = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
            prev, cum = d, torch.arange(d)
            for i, w in enumerate(self.cross_widths):
                assert w <= prev or i == 0
                self.V.append(nn.Linear(prev, rk[i], bias=False))
                self.U.append(nn.Linear(rk[i], w))
                rel = even_idx(prev, w)
                self.skip.append(nn.Identity() if w == prev else StridedSlice(rel))
                cum = cum if w == prev else cum[rel]
                self.anchor.append(nn.Identity() if w == d else StridedSlice(cum.clone()))
                prev = w
        self.parallel_dnn = MLP_Block(d, self.dnn_widths, "relu", None, None, net_dropout, batch_norm)
        out_w = self.cross_widths[-1] if self.cross_widths is not None else d
        self.fc = nn.Linear(out_w + self.dnn_widths[-1], 1)
        self.reset_parameters()
        if up_frozen:  # random orthogonal up-projection (orthonormal columns), frozen; only V and the bias train
            assert not full_rank
            for U in self.U:
                nn.init.orthogonal_(U.weight)
                U.weight.requires_grad_(False)

    def logits(self, feature_emb):
        x0 = self.proj(feature_emb.flatten(start_dim=1))
        x = x0
        if self.cross_widths is None:
            for V, U in zip(self.V, self.U):
                x = x + x0 * U(V(x))  # DCN V2 Eq. (2)
        else:
            for V, U, S, P in zip(self.V, self.U, self.skip, self.anchor):
                x = S(x) + P(x0) * U(V(x))
        return self.fc(torch.cat([x, self.parallel_dnn(x0)], dim=-1)).squeeze(-1)

    def analysis(self):
        """Effective rank exp(-sum p log p), p = sigma / sum sigma (Roy & Vetterli 2007), of each cross map U V."""
        out = []
        for V, U in zip(self.V, self.U):
            M = U.weight.detach().cpu().double()
            if not isinstance(V, nn.Identity):
                M = M @ V.weight.detach().cpu().double()
            s = torch.linalg.svdvals(M)
            p = s / s.sum()
            out.append(round(float(torch.exp(-(p * torch.log(p.clamp_min(1e-300))).sum())), 3))
        return {"eff_rank": out}

    def stacks(self):
        return {"cross": [self.d0, self.d, self.r, self.d], "dnn": [self.d] + self.dnn_widths}

    def width_checks(self):
        return []


class O256QNN(PQNN):
    def __init__(self, vocab_size, num_fields=39, emb_dim=16, d=256, num_layers=4, num_row=5, net_dropout=0.1):
        _FuxiCTRModel.__init__(self, vocab_size, num_fields, emb_dim)
        self.d0, self.widths, self.num_rows = num_fields * emb_dim, [d] * num_layers, [num_row] * num_layers
        self.qnn = nn.Sequential(nn.Linear(self.d0, d, bias=False),
                                 PQNNNet(d, self.widths, self.num_rows, net_dropout, False, "slice"))
        self.reset_parameters()

    def stacks(self):
        return {"qnn": [self.d0] + self.widths}

    def width_checks(self):
        return [(f"layer{i} rows*d_out <= d_in (R={r})", r * w, w) for i, (w, r) in
                enumerate(zip(self.widths, self.num_rows))]


class O256MaskNet(_FuxiCTRModel):
    def __init__(self, vocab_size, num_fields=39, emb_dim=16, d=256, num_blocks=2, reduction_ratio=2,
                 dnn_widths=(256,) * 5, net_dropout=0.3):
        super().__init__(vocab_size, num_fields, emb_dim)
        self.d0, self.d, self.dnn_widths = num_fields * emb_dim, d, list(dnn_widths)
        self.proj = nn.Linear(self.d0, d, bias=False)
        self.mask_net = xb.ParallelMaskNet(d, 1, num_blocks, d, self.dnn_widths, "relu", reduction_ratio,
                                           net_dropout, False)
        self.reset_parameters()

    def logits(self, feature_emb):
        V = self.proj(feature_emb.flatten(start_dim=1))
        return self.mask_net(V, V).squeeze(-1)

    def stacks(self):
        return {"in": [self.d0, self.d]}

    def width_checks(self):
        return []


class O256MLDCN(_FuxiCTRModel):
    """ML-DCN Eq. (8) on the projected stream: x <- LN(x0 * U((V x) * M(x)) + x), M: d -> t d -> r."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, d=256, num_layers=3, rank=None, t=0.5):
        super().__init__(vocab_size, num_fields, emb_dim)
        self.d0, self.d, self.r = num_fields * emb_dim, d, rank or d // 4
        r, k = self.r, int(t * d)
        self.proj = nn.Linear(self.d0, d, bias=False)
        self.V = nn.ModuleList(nn.Linear(d, r, bias=False) for _ in range(num_layers))
        self.U = nn.ModuleList(nn.Linear(r, d) for _ in range(num_layers))
        self.masks = nn.ModuleList(nn.Sequential(nn.Linear(d, k), nn.ReLU(), nn.Linear(k, r))
                                   for _ in range(num_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(num_layers))
        self.fc = nn.Linear(d, 1)
        self.reset_parameters()
        for m in self.masks:  # identity mask at init, as in the ML-DCN baseline of Table 2
            nn.init.zeros_(m[-1].weight)
            nn.init.ones_(m[-1].bias)

    def logits(self, feature_emb):
        x0 = self.proj(feature_emb.flatten(start_dim=1))
        x = x0
        for V, U, M, N in zip(self.V, self.U, self.masks, self.norms):
            x = N(x0 * U(V(x) * M(x)) + x)
        return self.fc(x).squeeze(-1)

    def stacks(self):
        return {"in": [self.d0, self.d]}

    def width_checks(self):
        return []


class O256ECN(_FuxiCTRModel):
    """Constant-width control: official ECN (4 layers, LN, dropout 0.1) on the projected d-wide stream."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, d=256, num_cross_layers=4, net_dropout=0.1):
        super().__init__(vocab_size, num_fields, emb_dim)
        self.d0, self.d = num_fields * emb_dim, d
        self.proj = nn.Linear(self.d0, d, bias=False)
        self.ECN = xb.ExponentialCrossNetwork(d, num_cross_layers, True, False, net_dropout, 1)
        self.reset_parameters()

    def logits(self, feature_emb):
        return self.ECN(self.proj(multihead_feature_emb(feature_emb, 1))).mean(dim=1).squeeze(-1)

    def stacks(self):
        return {"in": [self.d0, self.d]}

    def width_checks(self):
        return []


class O256DCNv3(xb.DCNv3):
    """Constant-width control: official DCNv3 (ECN 3 deep || LCN 4 shallow, Tri-BCE) on the projected stream."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, d=256, num_deep_cross_layers=3,
                 num_shallow_cross_layers=4, deep_net_dropout=0.1, shallow_net_dropout=0.2):
        _FuxiCTRModel.__init__(self, vocab_size, num_fields, emb_dim)
        self.num_heads, self.d0, self.d = 1, num_fields * emb_dim, d
        self.proj = nn.Linear(self.d0, d, bias=False)
        self.ECN = xb.ExponentialCrossNetwork(d, num_deep_cross_layers, True, False, deep_net_dropout, 1)
        self.LCN = xb.LinearCrossNetwork(d, num_shallow_cross_layers, True, False, shallow_net_dropout, 1)
        self.reset_parameters()

    def branch_logits(self, cats):
        x = self.proj(multihead_feature_emb(self.embed(cats), 1))
        return self.ECN(x).mean(dim=1).squeeze(-1), self.LCN(x).mean(dim=1).squeeze(-1)

    def stacks(self):
        return {"in": [self.d0, self.d]}

    def width_checks(self):
        return []


class PyramidLCNNet(nn.Module):
    """DCNv3 LinearCrossNetwork generalized to d_in -> d_out: H = W x (d_in -> d_out/2); z = [H, H*ReLU(LN(H))];
    x' = P(x0) * (z + b) + S(x).  P selects the same original coordinates as the composed skips (slice) or is a
    learned d0 -> d_out map (proj)."""

    def __init__(self, d0, widths, layer_norm=True, net_dropout=0.2, shrink="slice"):
        super().__init__()
        self.w, self.b, self.skip, self.anchor = nn.ModuleList(), nn.ParameterList(), nn.ModuleList(), nn.ModuleList()
        self.layer_norm, self.dropout = nn.ModuleList(), nn.ModuleList()
        prev, cum = d0, torch.arange(d0)
        for i, d in enumerate(widths):
            assert d % 2 == 0 and d <= prev, (prev, d)
            self.w.append(nn.Linear(prev, d // 2, bias=False))
            self.b.append(nn.Parameter(torch.empty((d,))))
            nn.init.uniform_(self.b[i].data)
            if layer_norm:
                self.layer_norm.append(nn.LayerNorm(d // 2))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))
            self.skip.append(make_shrink(prev, d, shrink))
            cum = cum[even_idx(prev, d)] if d != prev else cum
            self.anchor.append(make_shrink(d0, d, shrink, idx=cum.clone() if (shrink == "slice" and d != d0) else None))
            prev = d
        self.masker = nn.ReLU()
        self.fc = nn.Linear(prev, 1)

    def forward(self, x):
        x0 = x
        for i in range(len(self.w)):
            H = self.w[i](x)
            mask = self.masker(self.layer_norm[i](H)) if len(self.layer_norm) > i else self.masker(H)
            x = self.anchor[i](x0) * (torch.cat([H, H * mask], dim=-1) + self.b[i]) + self.skip[i](x)
            if len(self.dropout) > i:
                x = self.dropout[i](x)
        return self.fc(x)


class PDCNv3(xb.DCNv3):
    """DCNv3 with both branches narrowing (no map expands, widths non-increasing); Tri-BCE as the official."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, ecn_widths=(624,) * 3, lcn_widths=(624,) * 4,
                 deep_net_dropout=0.1, shallow_net_dropout=0.2, shrink="slice"):
        _FuxiCTRModel.__init__(self, vocab_size, num_fields, emb_dim)
        self.num_heads, self.d0 = 1, num_fields * emb_dim
        self.ecn_widths, self.lcn_widths = list(ecn_widths), list(lcn_widths)
        self.ECN = PyramidECNNet(self.d0, self.ecn_widths, True, False, deep_net_dropout, 1, shrink)
        self.LCN = PyramidLCNNet(self.d0, self.lcn_widths, True, shallow_net_dropout, shrink)
        self.reset_parameters()

    def stacks(self):
        return {"ecn": [self.d0] + self.ecn_widths, "lcn": [self.d0] + self.lcn_widths}

    def width_checks(self):
        out = []
        for name, ws in (("ecn", self.ecn_widths), ("lcn", self.lcn_widths)):
            prevs = [self.d0] + ws[:-1]
            out += [(f"{name} layer{i} concat [H, H*mask] <= d_in", d, p) for i, (p, d) in enumerate(zip(prevs, ws))]
        return out


# --------------------------------------------------------------------------- pyramids that share the original's first layer
# "Pyramid = the original with the same first layer and narrower later layers": the bias-free 624 -> d projection
# (the first layer of the projected originals O256*) is kept, the first cross layer keeps width d, and the later
# layers halve with parameter-free strided-slice skips/anchors (so the pyramid never costs more than the original).
class PP256DCNv2(_FuxiCTRModel):
    def __init__(self, vocab_size, num_fields=39, emb_dim=16, d=256, cross_widths=(256, 128, 64),
                 dnn_widths=(256, 128, 64, 32, 16), net_dropout=0.1, batch_norm=True, shrink="slice", res="state",
                 rr="proj"):
        super().__init__(vocab_size, num_fields, emb_dim)
        self.d0, self.d, self.cross_widths, self.dnn_widths = num_fields * emb_dim, d, list(cross_widths), list(dnn_widths)
        self.proj = nn.Linear(self.d0, d, bias=False)
        self.crossnet = PyramidCrossNet(d, self.cross_widths, shrink, res)
        self.parallel_dnn = make_mlp(d, self.dnn_widths, "relu", net_dropout, batch_norm, res, rr)
        self.fc = nn.Linear(self.cross_widths[-1] + self.dnn_widths[-1], 1)
        self.reset_parameters()

    def logits(self, feature_emb):
        x0 = self.proj(feature_emb.flatten(start_dim=1))
        return self.fc(torch.cat([self.crossnet(x0), self.parallel_dnn(x0)], dim=-1)).squeeze(-1)

    def stacks(self):
        return {"cross": [self.d, *self.cross_widths], "dnn": [self.d, *self.dnn_widths]}

    def width_checks(self):
        return []


class PP256QNN(PQNN):
    def __init__(self, vocab_size, num_fields=39, emb_dim=16, d=256, widths=(256, 128, 64, 32), num_rows=(1, 2, 2, 2),
                 net_dropout=0.1, shrink="slice", res="state", rr="proj", in_proj=True):
        _FuxiCTRModel.__init__(self, vocab_size, num_fields, emb_dim)
        self.d0, self.d, self.widths, self.num_rows = num_fields * emb_dim, d, list(widths), list(num_rows)
        assert in_proj or d == self.d0  # in_proj=False: read the input directly (no learned projection)
        self.qnn = nn.Sequential(nn.Linear(self.d0, d, bias=False) if in_proj else nn.Identity(),
                                 PQNNNet(d, self.widths, self.num_rows, net_dropout, False, shrink, res, rr))
        self.reset_parameters()

    def stacks(self):
        return {"qnn": [self.d, *self.widths]}

    def width_checks(self):
        prevs = [self.d] + self.widths[:-1]
        return [(f"layer{i} rows*d_out <= d_in (R={r})", r * w, p)
                for i, (p, w, r) in enumerate(zip(prevs, self.widths, self.num_rows))]


class PP256ECN(_FuxiCTRModel):
    def __init__(self, vocab_size, num_fields=39, emb_dim=16, d=256, widths=(256, 128, 64, 32), net_dropout=0.1,
                 shrink="slice", res="state", rr="proj", in_proj=True):
        super().__init__(vocab_size, num_fields, emb_dim)
        self.d0, self.d, self.widths = num_fields * emb_dim, d, list(widths)
        assert in_proj or d == self.d0  # in_proj=False: read the input directly (no learned projection)
        self.proj = nn.Linear(self.d0, d, bias=False) if in_proj else nn.Identity()
        self.ECN = PyramidECNNet(d, self.widths, True, False, net_dropout, 1, shrink, res, rr)
        self.reset_parameters()

    def logits(self, feature_emb):
        return self.ECN(self.proj(multihead_feature_emb(feature_emb, 1))).mean(dim=1).squeeze(-1)

    def stacks(self):
        return {"ecn": [self.d, *self.widths]}

    def width_checks(self):
        prevs = [self.d] + self.widths[:-1]
        return [(f"layer{i} concat <= d_in", w, p) for i, (p, w) in enumerate(zip(prevs, self.widths))]


class PP256DCNv3(xb.DCNv3):
    def __init__(self, vocab_size, num_fields=39, emb_dim=16, d=256, ecn_widths=(256, 128, 64),
                 lcn_widths=(256, 128, 64, 32), deep_net_dropout=0.1, shallow_net_dropout=0.2, shrink="slice"):
        _FuxiCTRModel.__init__(self, vocab_size, num_fields, emb_dim)
        self.num_heads, self.d0, self.d = 1, num_fields * emb_dim, d
        self.ecn_widths, self.lcn_widths = list(ecn_widths), list(lcn_widths)
        self.proj = nn.Linear(self.d0, d, bias=False)
        self.ECN = PyramidECNNet(d, self.ecn_widths, True, False, deep_net_dropout, 1, shrink)
        self.LCN = PyramidLCNNet(d, self.lcn_widths, True, shallow_net_dropout, shrink)
        self.reset_parameters()

    def branch_logits(self, cats):
        x = self.proj(multihead_feature_emb(self.embed(cats), 1))
        return self.ECN(x).mean(dim=1).squeeze(-1), self.LCN(x).mean(dim=1).squeeze(-1)

    def stacks(self):
        return {"ecn": [self.d, *self.ecn_widths], "lcn": [self.d, *self.lcn_widths]}

    def width_checks(self):
        return []


# --------------------------------------------------------------------------- presets + factory
def _halve(first, n):
    out = [first]
    for _ in range(n - 1):
        out.append(out[-1] // 2)
    return out


def preset_kwargs(name, preset, d0=624):
    """Width presets.  same: first width = min(official first width, d0), then halving.
    iso: max-FLOPs compliant config at official depth (all stacks at their widest allowed width); for p_ecn this
    is exactly the official model.  official: the official (non-compliant except ECN) widths, for reference."""
    if name == "p_dcnv2":
        return {"same": dict(cross_widths=_halve(d0, 3), dnn_widths=_halve(min(1000, d0), 5)),
                "iso": dict(cross_widths=[d0] * 3, dnn_widths=[d0] * 5),
                "official": dict(cross_widths=[d0] * 3, dnn_widths=[1000] * 5)}[preset]
    if name == "p_finalmlp":
        return {"same": dict(mlp1_widths=_halve(min(1000, d0), 5), mlp2_widths=[min(500, d0)]),
                "iso": dict(mlp1_widths=[d0] * 5, mlp2_widths=[min(500, d0)]),
                "official": dict(mlp1_widths=[1000] * 5, mlp2_widths=[500], num_heads=50)}[preset]
    if name == "p_masknet":
        return {"same": dict(block_dim=min(1000, d0) // 2, dnn_widths=_halve(min(1000, d0), 5)),
                "iso": dict(block_dim=d0 // 2, dnn_widths=[d0] * 5),
                "official": dict(block_dim=500, dnn_widths=[1000] * 5, mask_space="input",
                                 reduction_ratio=0.1)}[preset]
    if name == "p_qnn":
        return {"same": dict(widths=_halve(d0, 4)),  # rows = max allowed (<= 5): [1, 2, 2, 2]
                "same_r2": dict(widths=_halve(d0 // 2, 4), num_rows=2),
                "iso": dict(widths=[d0] * 4, num_rows=1),
                "official": dict(widths=[d0] * 4, num_rows=5)}[preset]
    if name == "p_ecn":
        return {"same": dict(widths=_halve(d0, 4)),
                "iso": dict(widths=[d0] * 4),
                "official": dict(widths=[d0] * 4)}[preset]
    raise KeyError(name)


# FLOPs status of the iso presets at d0 = 624 (verified in pyramid_check.py)
ISO_STATUS = {"p_dcnv2": "capped", "p_finalmlp": "capped", "p_masknet": "capped", "p_qnn": "capped",
              "p_ecn": "matched (official ECN is already R1/R2-compliant)"}
ORIGINAL = {"p_dcnv2": "dcnv2", "p_finalmlp": "finalmlp", "p_masknet": "masknet", "p_qnn": "qnn_alpha",
            "p_ecn": "ecn"}
_BUILDERS = {"p_dcnv2": PDCNv2, "p_finalmlp": PFinalMLP, "p_masknet": PMaskNet, "p_qnn": PQNN, "p_ecn": PECN,
             "o256_dcnv2": O256DCNv2, "o256_qnn": O256QNN, "o256_masknet": O256MaskNet,
             "o256_mldcn": O256MLDCN, "o256_ecn": O256ECN, "o256_dcnv3": O256DCNv3, "p_dcnv3": PDCNv3,
             "pp256_dcnv2": PP256DCNv2, "pp256_qnn": PP256QNN, "pp256_ecn": PP256ECN, "pp256_dcnv3": PP256DCNv3}
PYRAMIDS = tuple(_BUILDERS)


def make_first_mask(d_in, d_out, t):
    """Instance mask eta = W2 ReLU(W1 x + b1) + b2, W1: d_in -> floor(t d_in), W2: -> d_out (no map expands),
    initialized to the identity (W2 = 0, b2 = 1) so the masked pyramid starts as the unmasked one."""
    k = math.floor(t * d_in)
    assert d_out <= k <= d_in, (d_in, k, d_out)
    m = nn.Sequential(nn.Linear(d_in, k), nn.ReLU(), nn.Linear(k, d_out))
    nn.init.xavier_normal_(m[0].weight)
    nn.init.zeros_(m[0].bias)
    nn.init.zeros_(m[2].weight)
    nn.init.ones_(m[2].bias)
    return m


def _mask_mlp_first_layer(block, t):
    """MLP_Block / ReReadMLP: eta(x) multiplies the first Linear's output W_1 x + b_1 (before BN / activation)."""
    lin = block.mlp[0]
    block.fmask = make_first_mask(lin.in_features, lin.out_features, t)
    lin.register_forward_hook(lambda m, inp, out: out * block.fmask(inp[0]))


def attach_first_mask(model, key, t):
    """Q5: one instance mask in the first interaction layer of a pyramid (the layer's input -> its first width),
    multiplying that layer's interaction term; every later layer is unchanged.  Attached after construction, so the
    official reset_parameters never touches it."""
    if key in ("p_dcnv2", "pp256_dcnv2"):
        W = model.crossnet.cross_layers[0]
        model.crossnet.fmask = make_first_mask(W.in_features, W.out_features, t)
    elif key == "pp256_qnn":
        layer = model.qnn[1].layer[0]
        layer.fmask = make_first_mask(layer.linear[0].in_features, layer.d_out, t)
    elif key == "pp256_ecn":
        model.ECN.fmask = make_first_mask(model.ECN.w[0].in_features, model.ECN.b[0].numel(), t)
    elif key == "p_gdcn":
        W = model.cross_net.w[0]
        model.cross_net.fmask = make_first_mask(W.in_features, W.out_features, t)
    elif key == "p_fcn":
        model.fmask = make_first_mask(model.tower_a.w[0].in_features, model.tower_a.b[0].numel(), t)
    elif key == "p_finalmlp":
        _mask_mlp_first_layer(model.mlp1, t)
    elif key == "p_fibinetpp":
        _mask_mlp_first_layer(model.dnn, t)
    else:
        raise KeyError(f"no first-layer mask defined for {key}")


def build_pyramid(name: str, vocab_size: int, num_fields: int = 39, emb_dim: int = 16, widths="same", **kw):
    key = name.lower()
    if key not in _BUILDERS:
        raise KeyError(f"unknown pyramid baseline {name!r}; choose from {PYRAMIDS}")
    first_mask_t = kw.pop("first_mask_t", None)
    cfg = preset_kwargs(key, widths, num_fields * emb_dim) if isinstance(widths, str) else dict(widths)
    if key == "p_masknet" and cfg.get("mask_space", kw.get("mask_space", "input")) == "output" \
            and "mask_hidden" not in kw and isinstance(widths, str):
        # output-space mask hidden width k: smallest compliant (= block_dim) for "same", widest (= d0) for "iso"
        cfg["mask_hidden"] = num_fields * emb_dim if widths == "iso" else cfg["block_dim"]
    cfg.update(kw)
    model = _BUILDERS[key](vocab_size, num_fields, emb_dim, **cfg)
    model.pyramid_name, model.preset = key, widths if isinstance(widths, str) else "custom"
    if first_mask_t is not None:
        attach_first_mask(model, key, first_mask_t)
    return model


def audit(model):
    """R1/R2 audit.  Returns dict(ok, r1_ok, r2_ok, violations=[...], scalar_heads=[...], stacks={...})."""
    violations, heads = [], []
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            if m.out_features == 1:
                heads.append(n)
            elif m.out_features > m.in_features:
                violations.append(f"R1 Linear {n}: {m.in_features} -> {m.out_features}")
        if hasattr(m, "einsum_maps"):
            for en, i, o in m.einsum_maps():
                if o > i:
                    violations.append(f"R1 einsum {en}: {i} -> {o}")
    for desc, w, lim in model.width_checks():
        if w > lim:
            violations.append(f"R1 {desc}: {w} > {lim}")
    r1_ok = not violations
    stacks = model.stacks()
    for s, ws in stacks.items():
        if not _nonincreasing(ws):
            violations.append(f"R2 stack {s} not non-increasing: {ws}")
    r2_ok = not any(v.startswith("R2") for v in violations)
    return dict(ok=not violations, r1_ok=r1_ok, r2_ok=r2_ok, violations=violations, scalar_heads=heads,
                stacks=stacks)


# registers the FCN / GDCN / FiBiNet++ originals and pyramids (pyramid_extra.py) in _BUILDERS
from . import pyramid_extra  # noqa: E402,F401
