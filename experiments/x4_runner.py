"""Criteo_x4_001 (BARS / QNN paper protocol), multi-epoch, validation-monitored.

Data: data/criteo_x4/fuxi_001 (prep_criteo_x4.py): 39 categorical fields
(13 bucketized numerics + 26 categoricals), one shared table with per-field OOV.
Protocol (FuxiCTR BaseModel.fit, QNN_T26 Criteo config): Adam lr 1e-3, batch
10000, emb 16, (λ/2)||E||² with λ=1e-5 on the full table, grad-norm clip 10,
valid AUC each epoch, lr *= 0.1 on no improvement, stop after 2 such epochs,
best-valid checkpoint; test once on that checkpoint with --eval_test.

Models
  qnn_alpha : official QNN_T26 (salmon1802/QNN src/QNN_T26.py) incl. SE loss
  pml       : bench embedding -> PyramidMLDCN(624, dims, t) -> Linear head
  pml_se    : pml + the same SE loss (dropout passes self-distilled to mean)
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score

from . import flops  # noqa: E402


# The model core can be imported without downloading datasets. Training requires
# an explicit prepared directory, selected with --data-dir or CTR_DATA.
from .paths import resolve_data_dir

DATA = None
DATASET = None
META = None
N_FIELDS, EMB = 39, 16


def configure_metadata(metadata, dataset="synthetic"):
    """Configure model shapes from prepared metadata (also used for smoke tests)."""
    global META, DATASET, N_FIELDS, EMB
    if "fields" not in metadata or "vocab_size" not in metadata:
        raise ValueError("metadata needs fields and vocab_size")
    META = dict(metadata)
    DATASET = dataset
    N_FIELDS = len(META["fields"])
    EMB = int(os.environ.get("CTR_EMB", META.get("emb_dim", 16)))


def configure_data(data_dir=None):
    """Select the prepared dataset used by the original one-pass protocol."""
    global DATA
    path = resolve_data_dir(data_dir)
    with (path / "meta.json").open() as stream:
        metadata = json.load(stream)
    DATA = str(path)
    dataset = f"{path.parent.name}_{path.name.replace('fuxi_', '')}"
    configure_metadata(metadata, dataset)


if os.environ.get("CTR_DATA"):
    configure_data()


def load(split):
    if DATA is None:
        configure_data()
    cats = torch.from_numpy(np.asarray(np.load(f"{DATA}/{split}_cats.npy", mmap_mode="r")))
    y = torch.from_numpy(np.asarray(np.load(f"{DATA}/{split}_y.npy", mmap_mode="r")))
    return cats, y


class QuadraticLayer(nn.Module):
    """QNN_T26 QuadraticLayer, num_heads=1, batch_norm=False."""

    def __init__(self, d, num_row, dropout):
        super().__init__()
        self.linear = nn.Linear(d, d * num_row)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.num_row, self.d = num_row, d

    def forward(self, x):
        h = self.dropout(F.relu(self.linear(x)).view(-1, self.num_row, self.d))
        return (x.unsqueeze(1) * h).sum(dim=1) + x


class QNNAlpha(nn.Module):
    def __init__(self, num_layers=4, num_row=5, dropout=0.1):
        super().__init__()
        d = N_FIELDS * EMB
        self.layers = nn.ModuleList(QuadraticLayer(d, num_row, dropout) for _ in range(num_layers))
        self.dropouts = nn.ModuleList(nn.Dropout(dropout) for _ in range(num_layers))
        self.fc = nn.Linear(d, 1)

    def forward(self, x):
        for layer, drop in zip(self.layers, self.dropouts):
            x = drop(layer(x))
        return self.fc(x).squeeze(-1)


class PMLHead(nn.Module):
    """Unavailable legacy production branch; saved paper arms use pml_mv."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("The legacy production PyramidMLDCN branch is outside this release; use the published pml_mv arms.")


class CondHead(nn.Module):
    """cond_pyramid_screen.CondLayer stack (per-layer mask rows) + dropout after each LN."""

    def __init__(self, dims, rows, t, dropout=0.0, mask_src="state", inter_drop=0.0):
        super().__init__()
        from .cond_pyramid_screen import CondLayer

        d0 = N_FIELDS * EMB
        self.layers, self.norms, prev = nn.ModuleList(), nn.ModuleList(), d0
        for dim, r in zip(dims, rows):
            self.layers.append(CondLayer(prev, d0, dim, r, t, mask_src, inter_drop))
            self.norms.append(nn.LayerNorm(dim))
            prev = dim
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(prev, 1)

    def forward(self, x0):
        x = x0
        for layer, norm in zip(self.layers, self.norms):
            x = self.dropout(norm(layer(x0, x)))
        return self.fc(x).squeeze(-1)


class PyramidExpHead(nn.Module):
    """Narrow first, then go exponential.

    Linear-order pyramid (CondLayer stack, per-layer mask rows, LN) down to d_L,
    then `exp_layers` official ECN self-cross layers at width d_L
    (H = W x, d_L -> d_L/2; x = x ⊙ ([H, H ⊙ ReLU(LN(H))] + b) + x), whose order
    doubles per layer.  At the neck these cost d_L²/2 MACs each.
    """

    def __init__(self, dims, rows, t, exp_layers, exp_dropout=0.0, dropout=0.0, exp_identity=False):
        super().__init__()
        from .cond_pyramid_screen import CondLayer
        from .x4_baselines import ExponentialCrossNetwork

        d0 = N_FIELDS * EMB
        self.layers, self.norms, prev = nn.ModuleList(), nn.ModuleList(), d0
        for dim, r in zip(dims, rows):
            self.layers.append(CondLayer(prev, d0, dim, r, t))
            self.norms.append(nn.LayerNorm(dim))
            prev = dim
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.exp = ExponentialCrossNetwork(prev, exp_layers, layer_norm=True, net_dropout=exp_dropout)
        for m in self.exp.modules():  # FuxiCTR reset_parameters, as in the official ECN
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if exp_identity:  # W = b = 0: every exponential layer starts as the identity x -> x
            for w, b in zip(self.exp.w, self.exp.b):
                nn.init.zeros_(w.weight)
                nn.init.zeros_(b)

    def forward(self, x0):
        x = x0
        for layer, norm in zip(self.layers, self.norms):
            x = self.dropout(norm(layer(x0, x)))
        return self.exp(x).squeeze(-1)


class PMLECNHead(nn.Module):
    """Fast learner ‖ high-ceiling learner: PML pyramid branch + official ECN branch on the same x0.

    Served logit = mean of the two branch logits; trained with DCNv3's Tri-BCE
    (L + L_a·max(L_a − L, 0) + L_b·max(L_b − L, 0), weights not detached), i.e.
    DCNv3 with its linear-order LCN branch replaced by the PML pyramid.
    """

    def __init__(self, dims, rows, t, ecn_layers=3, ecn_dropout=0.1, dropout=0.0, ecn_bn=False):
        super().__init__()
        from .x4_baselines import ExponentialCrossNetwork

        self.pml = CondHead(dims, rows, t, dropout)
        self.ecn = ExponentialCrossNetwork(N_FIELDS * EMB, ecn_layers, layer_norm=True, batch_norm=ecn_bn,
                                           net_dropout=ecn_dropout)
        for m in self.ecn.modules():  # FuxiCTR reset_parameters, as in the official ECN/DCNv3
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def branch_logits(self, x0):
        # official ECN layout is B x heads x D (BatchNorm1d(num_heads) acts on dim 1); one head here
        return self.pml(x0), self.ecn(x0.unsqueeze(1)).reshape(-1)

    def forward(self, x0):
        a, b = self.branch_logits(x0)
        return 0.5 * (a + b)

    def tri_bce(self, x0, y, heads=1):
        a, b = (z.view(-1, heads).mean(dim=1) for z in self.branch_logits(x0))
        loss = F.binary_cross_entropy_with_logits(0.5 * (a + b), y)
        loss_a = F.binary_cross_entropy_with_logits(a, y)
        loss_b = F.binary_cross_entropy_with_logits(b, y)
        return loss + loss_a * (loss_a - loss).clamp_min(0) + loss_b * (loss_b - loss).clamp_min(0)


class PMLECNMultiViewHead(nn.Module):
    """PMLECNHead whose re-reads see *different* embedding views.

    The table is EMB*V wide and split into V views x0^(0..V-1). The ECN branch
    reads view 0; pyramid layer i re-reads view 1 + i mod (V-1) (layer 1 also
    takes that view as its state input). Embedding capacity grows V-fold while
    the dense FLOPs stay those of the single-view PMLECNHead.
    """

    def __init__(self, dims, rows, t, views, ecn_layers=4, ecn_dropout=0.0, dropout=0.0, ecn_bn=True):
        super().__init__()
        from .cond_pyramid_screen import CondLayer
        from .x4_baselines import ExponentialCrossNetwork

        d0 = N_FIELDS * EMB
        self.views = views
        self.layers, self.norms, prev = nn.ModuleList(), nn.ModuleList(), d0
        for dim, r in zip(dims, rows):
            self.layers.append(CondLayer(prev, d0, dim, r, t))
            self.norms.append(nn.LayerNorm(dim))
            prev = dim
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(prev, 1)
        self.ecn = ExponentialCrossNetwork(d0, ecn_layers, layer_norm=True, batch_norm=ecn_bn, net_dropout=ecn_dropout)
        for m in self.ecn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def view_of_layer(self, i):
        return 1 + i % (self.views - 1)

    def branch_logits(self, views):
        x = views[self.view_of_layer(0)]
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = self.dropout(norm(layer(views[self.view_of_layer(i)], x)))
        return self.fc(x).squeeze(-1), self.ecn(views[0].unsqueeze(1)).reshape(-1)

    def forward(self, views):
        a, b = self.branch_logits(views)
        return 0.5 * (a + b)

    def tri_bce(self, views, y):
        a, b = self.branch_logits(views)
        loss = F.binary_cross_entropy_with_logits(0.5 * (a + b), y)
        loss_a = F.binary_cross_entropy_with_logits(a, y)
        loss_b = F.binary_cross_entropy_with_logits(b, y)
        return loss + loss_a * (loss_a - loss).clamp_min(0) + loss_b * (loss_b - loss).clamp_min(0)


class PyramidMultiViewHead(nn.Module):
    """Pure re-reading pyramid (CondLayer stack) where layer i re-reads embedding view i.

    The table is EMB*V wide (V = number of layers); layer 1 takes view 0 both as its
    re-read input and as its state input, layer i>1 re-reads view i-1 and gates it with
    the previous state. Dense FLOPs equal the single-view pyramid; only embedding
    capacity grows.
    """

    def __init__(self, dims, rows, t, mask_src="state", anchor="learned", ln=True, reread="input", state_res=False,
                 res=True, mask_hidden=None, res_mode="same", field_dims=None, field_alpha=1.0, cross="dcn",
                 readout="last"):
        super().__init__()
        from .cond_pyramid_screen import CondLayer
        from .pyramid_baselines import StridedSlice, even_idx

        # ablations (CIKM'26 Table 3 style): ln=False drops LayerNorm; reread="state" makes layer i>1 re-read a
        # strided slice of the previous state instead of the input; reread="swap" also makes it cross the input instead of
        # the state, h = x~_{i-1} * (1 + A x + b) (roles of PyramidDCN swapped); state_res=True adds the carried state
        # (strided slice) as an extra residual; res=False drops the re-read residual (h = p * c).
        # state_res="learned": CIKM'26 projected hidden-state residual r_i = R_i x_{i-1}, R_i: d_{i-1} -> d_i, no bias.
        # mask_hidden: per-layer hidden width of the instance mask MLP (rows[i] > 0), overriding t.
        d0 = N_FIELDS * EMB
        self.reread = reread
        self.layers, self.norms, prev = nn.ModuleList(), nn.ModuleList(), d0
        self.skips = nn.ModuleList() if state_res else None
        # anchor="stagger": the copy re-read with selections disjoint across layers 2, 3, ...:
        # layer i >= 2 keeps every 2^(i-1)-th input coordinate starting at 0-based offset 2^(i-2) - 1, i.e. 1-based
        # 1,3,5,... | 2,6,10,... | 4,12,20,...; layer 1 (width d0) keeps all of them, as with anchor="copy"
        # anchor="disjoint": the general rule of pyramid_baselines.disjoint_idx (each layer avoids the coordinates kept by the
        # layers before it); identical to "stagger" for [W, W/2, W/4]
        # anchor="balanced": the least-used rule of pyramid_baselines.balanced_idx (minimum overlap, uniform usage; the final
        # PyramidDCN re-read); identical to "stagger" and "disjoint" for halving widths
        # anchor="field": the planned field split (pyramid_baselines.field_plan_idx) over field_dims; with one field
        # (the shared projection, default) it equals "balanced"
        if anchor in ("disjoint", "balanced", "field"):
            from .pyramid_baselines import disjoint_idx, balanced_idx, field_split_idx
            assert reread == "input", reread
            if anchor == "field":  # planned, disjoint field split (pyramid_baselines.field_plan_idx)
                from .pyramid_baselines import field_plan_idx
                fd = field_dims or [d0]
                assert sum(fd) == d0, (fd, d0)
                didx = field_plan_idx(fd, dims)
            else:
                didx = (disjoint_idx if anchor == "disjoint" else balanced_idx)(d0, dims)
        # cross: crossing form of every layer (cond_pyramid_screen.CondLayer); readout="concat" reads
        # every layer's state, the head being w^T [x_1; ...; x_L] + b (multi-layer readout)
        for i, (dim, r) in enumerate(zip(dims, rows)):  # mask_src="x0": the mask reads the layer's input view
            src = prev if (reread in ("state", "swap") and i) else d0
            off = (2 ** (i - 1) - 1) if (anchor == "stagger" and i >= 1) else 0
            a2 = None
            if cross == "two":  # the second selection: the first one shifted by half its stride (by d0/2 at full width)
                if anchor in ("disjoint", "balanced", "field"):
                    base_idx = [int(v) for v in didx[i]]
                else:
                    base_idx = list(range(d0)) if dim == d0 else [int(v) for v in (even_idx(d0, dim) + off)]
                stride = d0 // dim
                shift = stride // 2 if stride >= 2 else d0 // 2
                a2 = [(v + shift) % d0 for v in base_idx]
            self.layers.append(CondLayer(d0 if (cross == "inonly" or (reread == "swap" and i)) else prev, src, dim, r, t, mask_src=mask_src,
                                         anchor="copy" if anchor in ("stagger", "disjoint", "balanced", "field") else anchor,
                                         res=res, mask_hidden=mask_hidden[i] if mask_hidden else None, anchor_offset=off,
                                         res_mode=res_mode, anchor_idx=didx[i] if anchor in ("disjoint", "balanced", "field") else None,
                                         cross=cross, anchor2_idx=a2))
            self.norms.append(nn.LayerNorm(dim) if ln else nn.Identity())
            if state_res == "learned":
                from .cond_pyramid_screen import compat
                self.skips.append(nn.Linear(prev, dim, bias=False))
                compat.init_linear(self.skips[-1])
            elif state_res:
                self.skips.append(nn.Identity() if dim == prev else StridedSlice(even_idx(prev, dim)))
            prev = dim
        self.readout = readout
        self.fc = nn.Linear(sum(dims) if readout == "concat" else prev, 1)
        self.res_mode = res_mode

    def states(self, views):
        """x_1, ..., x_L (the forward pass, kept for analysis)."""
        x, out = views[0], []
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            v = x if (self.reread in ("state", "swap") and i) else views[i % len(views)]
            h = layer(v, views[0]) if (self.reread == "swap" and i) else layer(v, x)
            if self.skips is not None:
                h = h + self.skips[i](x)
            x = norm(h) + layer.proj_x0(v) if self.res_mode == "post" else norm(h)
            out.append(x)
        return out

    def forward(self, views):
        st = self.states(views)
        return self.fc(torch.cat(st, dim=-1) if self.readout == "concat" else st[-1]).squeeze(-1)


class MLDCNHead(nn.Module):
    """ML-DCN (arXiv:2602.09194, Eq. 8): X_{l+1} = LN(X0 ⊙ (((X_l V_l) ⊙ M_l) U_lᵀ + b_l) + X_l).

    M_l = MLP(X_l): d -> k = t·d -> r (ReLU hidden, linear output), applied in the
    rank-r space.  The paper's Eq. 5 gives a d-wide mask, inconsistent with Eq. 8;
    we use width r.  Identity mask init (as in PML) for a like-for-like comparison.
    """

    def __init__(self, rank, num_layers, t=0.5):
        super().__init__()
        d = N_FIELDS * EMB
        self.V = nn.ModuleList(nn.Linear(d, rank, bias=False) for _ in range(num_layers))
        self.U = nn.ModuleList(nn.Linear(rank, d) for _ in range(num_layers))
        self.masks = nn.ModuleList(
            nn.Sequential(nn.Linear(d, int(t * d)), nn.ReLU(), nn.Linear(int(t * d), rank)) for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(num_layers))
        for m in self.masks:
            nn.init.zeros_(m[-1].weight)
            nn.init.ones_(m[-1].bias)
        self.fc = nn.Linear(d, 1)

    def forward(self, x0):
        x = x0
        for V, U, mask, norm in zip(self.V, self.U, self.masks, self.norms):
            x = norm(x0 * U(V(x) * mask(x)) + x)
        return self.fc(x).squeeze(-1)


class PyramidMLDCNHead(nn.Module):
    """ML-DCN's own layer with narrowing widths:
        x_l = LN(A_l(x0) * (U_l((V_l x_{l-1}) * M_l(x_{l-1})) + b_l) + S_l(x_{l-1})),
    V_l: d_{l-1} -> r_l, M_l: d_{l-1} -> int(t d_{l-1}) -> r_l (identity init), U_l: r_l -> d_l,
    A_l: the input re-read to width d_l (Identity when d_l = d0), S_l: the state shortcut cut to width d_l
    (Identity when equal; a strided slice when narrowing; a learned map only when the first layer widens the
    input).  dims = [d0]*L and ranks = [r]*L reproduce MLDCNHead exactly (same parameters, same init order)."""

    def __init__(self, dims, ranks, t=0.5):
        super().__init__()
        d0 = N_FIELDS * EMB
        prevs = [d0] + list(dims[:-1])
        self.V = nn.ModuleList(nn.Linear(p, r, bias=False) for p, r in zip(prevs, ranks))
        self.U = nn.ModuleList(nn.Linear(r, d) for r, d in zip(ranks, dims))
        self.masks = nn.ModuleList(nn.Sequential(nn.Linear(p, int(t * p)), nn.ReLU(), nn.Linear(int(t * p), r))
                                   for p, r in zip(prevs, ranks))
        self.norms = nn.ModuleList(nn.LayerNorm(d) for d in dims)
        for m in self.masks:
            nn.init.zeros_(m[-1].weight)
            nn.init.ones_(m[-1].bias)
        self.fc = nn.Linear(dims[-1], 1)
        self.anchor = nn.ModuleList(nn.Identity() if d == d0 else  # wider than the input: parameter-free copy of x0
                                    _Slice(torch.arange(d) * d0 // d) if d > d0 else nn.Linear(d0, d, bias=False)
                                    for d in dims)
        self.skip = []
        for p, d in zip(prevs, dims):
            if d == p:
                self.skip.append(nn.Identity())
            elif d < p:
                self.skip.append(_Slice(torch.linspace(0, p - 1, d).round().long()))
            else:  # the first layer widens the input: its residual is the input re-read A_l(x0) (no extra map)
                self.skip.append(None)
        self._skip_mods = nn.ModuleList(m for m in self.skip if m is not None)  # registers the slice buffers

    def forward(self, x0):
        x = x0
        for V, U, mask, norm, A, S in zip(self.V, self.U, self.masks, self.norms, self.anchor, self.skip):
            a = A(x0)
            x = norm(a * U(V(x) * mask(x)) + (a if S is None else S(x)))
        return self.fc(x).squeeze(-1)


class _Slice(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.register_buffer("idx", idx, persistent=False)

    def forward(self, x):
        return x.index_select(-1, self.idx)


class PyramidHead(nn.Module):
    """Unavailable legacy production branch; saved paper arms use pml_mv."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("The legacy production PyramidDCN branch is outside this release; use the published pml_mv arms.")


class ProjEmbedding(nn.Module):
    """Protocol v3 shared input stage: one EMB-wide table, flattened (d0), then a bias-free Linear d0 -> W
    (xavier normal, as FuxiCTR initializes Linear layers).  Every backbone sees the same W-wide input, viewed as
    W/8 pseudo-fields of width 8 (models that only flatten their input are unaffected by the split)."""

    def __init__(self, vocab_size, num_fields, emb_dim, width, pe=8):
        super().__init__()
        assert width % pe == 0, width
        self.table = nn.Embedding(vocab_size, emb_dim)
        nn.init.normal_(self.table.weight, std=1e-4)
        self.proj = nn.Linear(num_fields * emb_dim, width, bias=False)
        nn.init.xavier_normal_(self.proj.weight)
        self.pf, self.pe = width // pe, pe

    @property
    def weight(self):  # the runner treats model.embedding.weight as the embedding table
        return self.table.weight

    def forward(self, cats):
        return self.proj(self.table(cats.long()).flatten(1)).view(-1, self.pf, self.pe)


class CTRModel(nn.Module):
    def __init__(self, arm):
        super().__init__()
        if META is None:
            configure_data()
        width = arm.get("input_proj")
        if not width:
            self._init(arm)
            return
        global N_FIELDS, EMB
        saved = (N_FIELDS, EMB)
        N_FIELDS, EMB = width // 8, 8  # build every backbone on the same W-wide projected input
        try:
            self._init(arm)
        finally:
            N_FIELDS, EMB = saved
        pe = ProjEmbedding(META["vocab_size"], saved[0], saved[1], width)
        if self.base is not None:
            self.base.embedding = pe
        else:
            assert self.views == 1 and self.heads == 1 and not self.field_ln and self.se is None
        self.embedding = pe

    def _init(self, arm):
        if arm["model"].startswith("pyramid:"):
            from .pyramid_baselines import build_pyramid

            self.base = build_pyramid(arm["model"].split(":", 1)[1], META["vocab_size"], N_FIELDS, EMB,
                                      widths=arm.get("preset", "same"), **arm.get("kw", {}))
            self.embedding = self.base.embedding
            return
        if arm["model"].startswith("baseline:"):
            from .x4_baselines import build_baseline
            from .extra_baselines import EXTRA_BASELINES, build_extra

            name = arm["model"].split(":", 1)[1]
            builder = build_extra if name in EXTRA_BASELINES else build_baseline
            self.base = builder(name, META["vocab_size"], N_FIELDS, EMB, **arm.get("overrides", {}))
            self.embedding = self.base.embedding
            return
        self.base = None
        # FCN-style multi-head embedding: each field's EMB*heads vector is split into `heads`
        # EMB-wide views; the network runs on every head with shared weights, logits averaged
        self.heads = arm.get("heads", 1)
        self.views = arm.get("views", 1) if arm["model"] in {"pml_ecn_mv", "pml_mv", "pml_gate"} else 1
        self.embedding = nn.Embedding(META["vocab_size"], EMB * self.heads * self.views)
        nn.init.normal_(self.embedding.weight, std=arm.get("emb_std", 1e-4))
        self.se = None
        if arm.get("se"):  # field-wise SENet+ re-weighting of every view (squeeze 2F*g -> r -> F, never expands)
            from .pyramid_extra import SENetFieldwise

            assert self.views > 1
            self.se = nn.ModuleList(SENetFieldwise(N_FIELDS, EMB) for _ in range(self.views))
            for m in self.se.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
        if arm["model"] == "mldcn":
            self.net = MLDCNHead(arm.get("rank", 256), arm.get("layers", 3), arm.get("t", 0.5))
        elif arm["model"] == "mldcn_pyr":
            self.net = PyramidMLDCNHead(arm["dims"], arm["ranks"], arm.get("t", 0.5))
        elif arm["model"] == "pyramid":
            self.net = PyramidHead(arm.get("dims", [256, 128, 64]))
        elif arm["model"] == "qnn_alpha":
            self.net = QNNAlpha(arm.get("layers", 4), arm.get("rows", 5), arm.get("dropout", 0.1))
            for m in self.net.modules():  # FuxiCTR reset_parameters
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)
        elif arm["model"] == "pml_res":
            raise NotImplementedError("Legacy pml_res is outside the paper experiment release.")

        elif arm["model"] == "pml_gate":
            raise NotImplementedError("Legacy pml_gate is outside the paper experiment release.")

        elif arm["model"] == "pml_mv":
            self.net = PyramidMultiViewHead(arm["dims"], arm["rows"], arm.get("t", 0.75), arm.get("mask_src", "state"),
                                            arm.get("anchor", "learned"), arm.get("ln", True), arm.get("reread", "input"),
                                            arm.get("state_res", False), arm.get("res", True), arm.get("mask_hidden"),
                                            arm.get("res_mode", "same"), arm.get("field_dims"), arm.get("field_alpha", 1.0),
                                            cross=arm.get("cross", "dcn"), readout=arm.get("readout", "last"))
        elif arm["model"] == "pml_ecn_mv":
            self.net = PMLECNMultiViewHead(arm["dims"], arm["rows"], arm.get("t", 0.75), arm["views"],
                                           arm.get("ecn_layers", 4), arm.get("ecn_dropout", 0.0),
                                           arm.get("dropout", 0.0), arm.get("ecn_bn", True))
        elif arm["model"] == "pml_ecn":
            self.net = PMLECNHead(arm["dims"], arm["rows"], arm.get("t", 0.75), arm.get("ecn_layers", 3),
                                  arm.get("ecn_dropout", 0.1), arm.get("dropout", 0.0), arm.get("ecn_bn", False))
        elif arm["model"] in {"px", "px_se"}:
            self.net = PyramidExpHead(arm["dims"], arm["rows"], arm.get("t", 0.75), arm["exp_layers"],
                                      arm.get("exp_dropout", 0.0), arm.get("dropout", 0.0),
                                      arm.get("exp_identity", False))
        elif arm["model"] in {"cond", "cond_se"}:
            self.net = CondHead(arm["dims"], arm["rows"], arm.get("t", 0.75), arm.get("dropout", 0.0),
                                arm.get("mask_src", "state"), arm.get("inter_drop", 0.0))
        else:
            self.net = PMLHead(arm["dims"], arm.get("t", 0.75), arm.get("dropout", 0.0))
        self.se_loss = arm["model"] in {"qnn_alpha", "pml_se", "cond_se", "px_se"}
        # per-field LayerNorm on the 39 field embeddings (affine per field), as MaskNet's emb_layernorm
        self.field_ln = arm.get("field_ln", False)
        if self.field_ln:
            self.ln_weight = nn.Parameter(torch.ones(N_FIELDS, EMB))
            self.ln_bias = nn.Parameter(torch.zeros(N_FIELDS, EMB))

    def embed(self, cats):
        e = self.embedding(cats)
        if self.heads > 1:  # B x F x (H*E) -> (B*H) x (F*E)
            e = e.view(*cats.shape, self.heads, EMB).transpose(1, 2).reshape(-1, cats.shape[1], EMB)
        if self.field_ln:
            e = F.layer_norm(e, (EMB,)) * self.ln_weight + self.ln_bias
        return e.flatten(1)

    def embed_views(self, cats):
        e = self.embedding(cats)
        if self.se is not None:
            return [self.se[v](e[..., v * EMB : (v + 1) * EMB]).flatten(1) for v in range(self.views)]
        return [e[..., v * EMB : (v + 1) * EMB].flatten(1) for v in range(self.views)]

    def forward(self, dense, cats):
        if self.base is not None:
            return self.base(dense, cats)
        if self.views > 1 or isinstance(self.net, PyramidMultiViewHead):  # pml_mv always takes a list of views
            return self.net(self.embed_views(cats))
        return self.net(self.embed(cats)).view(-1, self.heads).mean(dim=1)

    def loss(self, cats, y):
        if self.base is not None:
            return self.base.loss(cats, y)
        if isinstance(self.net, PMLECNMultiViewHead):
            return self.net.tri_bce(self.embed_views(cats), y)
        if isinstance(self.net, PMLECNHead):
            return self.net.tri_bce(self.embed(cats), y, self.heads)
        if not self.se_loss:
            return F.binary_cross_entropy_with_logits(self(None, cats), y)
        assert self.heads == 1, "SE loss with multi-head embeddings is not implemented"
        x = self.embed(cats)
        p1, p2 = torch.sigmoid(self.net(x)), torch.sigmoid(self.net(x))
        p = 0.5 * (p1 + p2)
        target = p.detach()
        return F.binary_cross_entropy(p, y) + F.binary_cross_entropy(p1, target) + F.binary_cross_entropy(p2, target)


@torch.no_grad()
def evaluate(model, cats, y, batch, device):
    model.eval()
    logits = [model(None, cats[i : i + batch].to(device)).float().cpu() for i in range(0, len(y), batch)]
    p = torch.sigmoid(torch.cat(logits)).double().numpy()
    return roc_auc_score(y.numpy(), p), log_loss(y.numpy(), np.clip(p, 1e-7, 1 - 1e-7))


def run_arm(arm, data, out, eval_test, device):
    (tr_cats, tr_y), (va_cats, va_y) = data
    if arm.get("train_frac", 1.0) < 1.0:  # x4 is a random split, so a prefix is a random subsample
        n = int(len(tr_y) * arm["train_frac"])
        tr_cats, tr_y = tr_cats[:n], tr_y[:n]
    seed = arm["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CTRModel(arm)
    if arm.get("emb_init_std") is not None:  # fair protocol: one embedding init for every model
        nn.init.normal_(model.embedding.weight, 0.0, arm["emb_init_std"])
    macs = flops.serving_linear_macs(model, torch.zeros(2, 0), tr_cats[:2])
    model = model.to(device)
    lr = arm.get("lr", 1e-3)
    emb_params = [model.embedding.weight]
    dense_params = [p for p in model.parameters() if p is not model.embedding.weight]
    opt = torch.optim.Adam([{"params": emb_params, "lr": lr * arm.get("emb_lr_mult", 1.0)},
                            {"params": dense_params, "lr": lr}])
    batch, patience, emb_reg = arm.get("batch", 10_000), 2, arm.get("emb_reg", 1e-5)
    sched = None
    if arm.get("sched") == "cosine":
        total = math.ceil(len(tr_y) / batch) * arm.get("max_epochs", 30)
        floor = arm.get("final_ratio", 0.01)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    best_auc, best_state, stop_steps, history, t_start = -1.0, None, 0, [], time.time()
    for epoch in range(1, arm.get("max_epochs", 30) + 1):
        model.train()
        perm = torch.randperm(len(tr_y), generator=torch.Generator().manual_seed(seed * 1000 + epoch))
        t0, losses = time.time(), []
        for i in range(0, len(tr_y), batch):
            idx = perm[i : i + batch]
            loss = model.loss(tr_cats[idx].to(device), tr_y[idx].to(device))
            if arm.get("nan_trace") and (i // batch) % 50 == 0:
                if not torch.isfinite(loss):
                    print(f"[{arm['name']}] NON-FINITE loss at step {i // batch}", flush=True)
                    raise FloatingPointError(f"non-finite loss at step {i // batch}")
                big = max(float(p.detach().abs().max()) for p in model.parameters())
                print(f"[{arm['name']}] step {i // batch} loss {loss.item():.4f} max|param| {big:.3g}", flush=True)
            opt.zero_grad()
            loss.backward()
            if emb_reg:
                model.embedding.weight.grad.add_(model.embedding.weight.detach(), alpha=emb_reg)
            if arm.get("clip", 10.0):
                nn.utils.clip_grad_norm_(model.parameters(), arm.get("clip", 10.0))
            opt.step()
            if sched is not None:
                sched.step()
            losses.append(loss.item())
        auc, ll = evaluate(model, va_cats, va_y, batch, device)
        history.append(dict(epoch=epoch, train_loss=round(float(np.mean(losses)), 6), valid_auc=round(auc, 6),
                            valid_logloss=round(ll, 6), lr=opt.param_groups[0]["lr"], epoch_s=round(time.time() - t0)))
        print(f"[{arm['name']}] " + json.dumps(history[-1]), flush=True)
        if auc > best_auc + 1e-6:
            best_auc, best_ll, best_epoch, stop_steps = auc, ll, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stop_steps += 1
            for g in opt.param_groups:
                g["lr"] = max(g["lr"] * 0.1, 1e-6)
            if stop_steps >= patience:
                break
    rec = dict(arm, dataset=DATASET, best_epoch=best_epoch, valid_auc=round(best_auc, 6),
               valid_logloss=round(best_ll, 6), serving_linear_macs=macs, serving_linear_flops_2x=2 * macs,
               dense_params=sum(p.numel() for n, p in model.named_parameters() if not n.startswith("embedding")),
               total_s=round(time.time() - t_start), history=history)
    if eval_test:
        model.load_state_dict(best_state)
        rec["test_auc"], rec["test_logloss"] = (round(v, 6) for v in evaluate(model, *load("test"), batch, device))
    analysis = getattr(getattr(model, "base", None), "analysis", None)  # e.g. effective rank of DCNv2 cross maps
    if analysis is not None:
        rec["analysis"] = analysis()
    if arm.get("funnel"):  # Thm. 2.1/2.2 check on the trained low-rank DCNv2 (float64, validation inputs)
        from .funnel_check import funnel_check
        rec["funnel"] = funnel_check(model, va_cats)
    if arm.get("erank"):  # effective rank of PyramidDCN layer outputs and crossing weights (validation inputs)
        from .erank_check import erank_check
        rec["erank"] = erank_check(model, va_cats)
    if arm.get("agop"):  # soft funnel (Theorem 1(c)): effective rank of E[g g^T] per Linear output, any model
        from .agop_check import agop_check
        rec["agop"] = agop_check(model, va_cats)
    print("RESULT " + json.dumps(rec), flush=True)
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arms", required=True, help="JSON arm list, or path to a JSON file")
    p.add_argument("--data-dir", help="Prepared dataset directory (defaults to CTR_DATA)")
    p.add_argument("--device", default="mps")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--eval_test", action="store_true")
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "x4_results.jsonl"))
    a = p.parse_args()
    configure_data(a.data_dir)
    torch.set_num_threads(a.threads)
    data = (load("train"), load("valid"))
    from pathlib import Path
    arm_path = Path(a.arms) if not a.arms.lstrip().startswith("[") else None
    arms = json.loads(arm_path.read_text() if arm_path is not None else a.arms)
    failures = []
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        try:
            run_arm(arm, data, a.out, a.eval_test, a.device)
        except Exception as exc:  # keep the queue alive; the failure is logged for diagnosis
            print(f"ARM FAILED [{arm['name']}]: {type(exc).__name__}: {exc}", flush=True)
            failures.append(arm["name"])
    if failures:
        raise SystemExit(f"{len(failures)} arm(s) failed: {failures}")
