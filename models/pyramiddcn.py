"""PyramidDCN (Eq. 1 of the paper) and the layer variants of Table 3.

Layer i reads a fixed slice p = x[S_i] of the projected input x and the previous state x_{i-1}:
    h_i = p * (A_i x_{i-1} + b_i) + p = p * (1 + A_i x_{i-1} + b_i),    x_i = LN(h_i),
with widths [W, W/2, W/4] by default.  The head is f = w^T x_L + b_o.

Table 3 variants (keyword arguments of PyramidDCN):
    anchor  "stagger" / "balanced"  fixed least-used channel selection (identical for halving widths; PyramidDCN)
            "copy"                   evenly spaced channels from offset 0, nested across layers ("Channel overlap")
            "pool"                   blockwise sum pooling of x, scaled by sqrt(d_i / W) ("Input pooling")
            "random" / "learned"     frozen random / learned projection of x ("Frozen / Learned projection")
    reread  "state"                  layers i > 1 select channels of the previous output ("Previous-layer channels")
    res     False                    h_i = p * (A_i x_{i-1} + b_i) ("No input skip")
    res_mode "comp" / "learned"      the added term is the selection shifted by half its stride / a learned projection
                                     ("Shifted input skip", "Projected input skip")
    cross   "add"                    h_i = (A_i x_{i-1} + b_i) + p ("Addition only")
            "inaware"                h_i = p * (1 + A_i x_{i-1} + B_i p + b_i) ("Input self-interaction")
Table 4 varies the widths (dims).
"""
import math

import torch.nn as nn

from .layers import StridedSlice, even_idx


def init_linear(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def balanced_idx(d0, dims):
    """Least-used selection rule of Section 3.1: layer i keeps the d_i input coordinates that layers 1..i-1 used least,
    evenly spaced among equally used ones (a layer wider than the input first takes every coordinate once, then
    continues with the least used).  For halving widths it gives 1,3,5,... | 2,6,10,... (1-based)."""
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


class GroupPool(nn.Module):
    """p_j = (x_{kj} + ... + x_{kj+k-1}) / sqrt(k): fixed, uses all input coordinates, keeps unit variance."""

    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        return x.unflatten(-1, (-1, self.k)).sum(-1) / math.sqrt(self.k)


class PyramidCrossLayer(nn.Module):
    """One cross layer before LayerNorm (see the module docstring for the variants)."""

    def __init__(self, in_dim, x0_dim, out_dim, anchor="copy", res=True, anchor_offset=0, res_mode="same",
                 anchor_idx=None, cross="dcn"):
        super().__init__()
        self.res = res
        if anchor == "copy":  # parameter-free slice of x (Identity at full width)
            if anchor_idx is not None:  # explicit coordinates (least-used rule)
                assert len(anchor_idx) == out_dim and max(anchor_idx) < x0_dim and res_mode != "comp", \
                    (out_dim, x0_dim, res_mode)
                self.proj_x0 = nn.Identity() if list(anchor_idx) == list(range(x0_dim)) else StridedSlice(anchor_idx)
            elif out_dim == x0_dim:
                self.proj_x0 = nn.Identity()
            else:  # anchor_offset shifts the strided selection (0-based); it must stay below the stride
                idx = even_idx(x0_dim, out_dim) + anchor_offset
                assert int(idx.max()) < x0_dim, (x0_dim, out_dim, anchor_offset)
                self.proj_x0 = StridedSlice(idx)
        elif anchor == "pool":
            assert out_dim <= x0_dim and x0_dim % out_dim == 0, (x0_dim, out_dim)
            self.proj_x0 = nn.Identity() if out_dim == x0_dim else GroupPool(x0_dim // out_dim)
        else:
            assert anchor in ("random", "learned"), anchor
            self.proj_x0 = nn.Linear(x0_dim, out_dim, bias=False)
        self.res_mode = res_mode
        if res_mode == "comp":  # the selection shifted by half its stride
            assert anchor == "copy" and res, "comp residual is defined for the parameter-free slice"
            if out_dim == x0_dim:
                self.proj_res = nn.Identity()
            else:
                stride = x0_dim // out_dim
                ridx = even_idx(x0_dim, out_dim) + anchor_offset + stride // 2
                assert x0_dim % out_dim == 0 and int(ridx.max()) < x0_dim, (x0_dim, out_dim)
                self.proj_res = StridedSlice(ridx)
        elif res_mode == "learned":  # bias-free learned map of x; the multiplier stays the fixed slice
            self.proj_res = nn.Linear(x0_dim, out_dim, bias=False)
        else:
            assert res_mode == "same", res_mode
        self.cross_mode = cross
        self.cross = nn.ModuleList([nn.Linear(in_dim, out_dim)])  # A_i x_{i-1} + b_i
        if cross != "dcn":
            assert res and res_mode == "same", (cross, res, res_mode)
            if cross == "inaware":
                self.cross2 = nn.Linear(out_dim, out_dim, bias=False)  # B_i
            else:
                assert cross == "add", cross
        self.apply(init_linear)
        if anchor == "random":
            self.proj_x0.weight.requires_grad_(False)

    def forward(self, x0, x):
        p = self.proj_x0(x0)
        c = self.cross[0](x)
        if self.cross_mode == "add":
            return c + p
        if self.cross_mode == "inaware":
            return p * (1 + c + self.cross2(p))
        if self.res_mode != "same":
            return p * c + self.proj_res(x0)
        return p * ((1 if self.res else 0) + c)


class PyramidDCN(nn.Module):
    """Stack of PyramidCrossLayer + LayerNorm on the W-wide projected input, then a linear head."""

    def __init__(self, d0, dims, anchor="stagger", reread="input", res=True, res_mode="same", cross="dcn"):
        super().__init__()
        assert reread in ("input", "state"), reread
        self.reread = reread
        self.layers, self.norms, prev = nn.ModuleList(), nn.ModuleList(), d0
        if anchor == "balanced":
            assert reread == "input", reread
            didx = balanced_idx(d0, dims)
        for i, dim in enumerate(dims):
            src = prev if (reread == "state" and i) else d0
            # "stagger": layer i >= 2 keeps every 2^(i-1)-th coordinate from 0-based offset 2^(i-2) - 1, i.e. 1-based
            # 1,3,5,... | 2,6,10,...; the first layer (width W) keeps all of them
            off = (2 ** (i - 1) - 1) if (anchor == "stagger" and i >= 1) else 0
            self.layers.append(PyramidCrossLayer(prev, src, dim,
                                                 anchor="copy" if anchor in ("stagger", "balanced") else anchor,
                                                 res=res, anchor_offset=off, res_mode=res_mode,
                                                 anchor_idx=didx[i] if anchor == "balanced" else None, cross=cross))
            self.norms.append(nn.LayerNorm(dim))
            prev = dim
        self.fc = nn.Linear(prev, 1)

    def states(self, x0):
        """x_1, ..., x_L."""
        x, out = x0, []
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            v = x if (self.reread == "state" and i) else x0
            x = norm(layer(v, x))
            out.append(x)
        return out

    def forward(self, x0):
        return self.fc(self.states(x0)[-1]).squeeze(-1)
