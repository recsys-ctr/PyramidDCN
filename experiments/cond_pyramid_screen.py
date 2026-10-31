"""Throwaway screen: where should instance-conditional capacity live in a pyramid?

Layer i (x0-residual PyramidMLDCN form, no up-projection anywhere):
    rows[i] == 0 : x_i = LN(P_i x0 ⊙ (1 + A_i x))                    (PyramidDCN)
    rows[i] == R : x_i = LN(P_i x0 ⊙ (1 + Σ_r m_ir(x) ⊙ A_ir x))     (R masked crosses)
m_ir = Linear(d_{i-1}, k) -> ReLU -> Linear(k, d_i), k = int(t d_{i-1}), identity init.
rows=(1,1) is exactly PyramidMLDCN [256,128] t=.75.  Same loop as evidence_screen.
"""
import math

import torch
import torch.nn as nn

from . import compat


class GroupPool(nn.Module):
    """p_j = (x_{kj} + ... + x_{kj+k-1}) / sqrt(k): fixed, uses all input coordinates, keeps unit variance."""

    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        return x.unflatten(-1, (-1, self.k)).sum(-1) / math.sqrt(self.k)


class CondLayer(nn.Module):
    def __init__(self, in_dim, x0_dim, out_dim, rows, t, mask_src="state", inter_drop=0.0, anchor="learned", res=True,
                 mask_hidden=None, anchor_offset=0, res_mode="same", anchor_idx=None, cross="dcn", anchor2_idx=None):
        super().__init__()
        self.mask_src, self.res = mask_src, res  # res=False drops the re-read residual: h = p * c (ablation)
        # interaction dropout: drops only the cross/interaction term, never the x0 view p
        self.inter_drop = nn.Dropout(inter_drop) if inter_drop > 0 else nn.Identity()
        m_in = x0_dim if mask_src == "x0" else in_dim
        if anchor == "copy":  # parameter-free re-read: copy (out > in) or strided selection (out < in) of x0
            from .pyramid_baselines import StridedSlice, even_idx
            # anchor_offset shifts the strided selection (0-based); it must stay below the stride
            if anchor_idx is not None:  # explicit coordinates (anchor="disjoint" in x4_runner, 2026-10-29)
                assert len(anchor_idx) == out_dim and max(anchor_idx) < x0_dim and res_mode not in ("comp",), (out_dim, x0_dim, res_mode)
                self.proj_x0 = nn.Identity() if list(anchor_idx) == list(range(x0_dim)) else StridedSlice(anchor_idx)
            elif out_dim == x0_dim:
                self.proj_x0 = nn.Identity()
            else:
                idx = even_idx(x0_dim, out_dim) + anchor_offset
                assert int(idx.max()) < x0_dim, (x0_dim, out_dim, anchor_offset)
                self.proj_x0 = StridedSlice(idx)
        elif anchor == "pool":  # parameter-free re-read using every coordinate: variance-preserving sum pooling
            assert out_dim <= x0_dim and x0_dim % out_dim == 0, (x0_dim, out_dim)
            self.proj_x0 = nn.Identity() if out_dim == x0_dim else GroupPool(x0_dim // out_dim)
        elif anchor == "random":  # the learned projection's map, frozen at its init
            self.proj_x0 = nn.Linear(x0_dim, out_dim, bias=False)
        else:
            assert anchor == "learned", anchor
            self.proj_x0 = nn.Linear(x0_dim, out_dim, bias=False)
        # res_mode):
        #   "gain" h = p * c + alpha * p with a learned per-unit residual gain alpha (init 1, out_dim parameters)
        #   "comp" h = p * c + r, r the complementary selection of x (the strided one shifted by half the stride)
        #   "post" h = p * c here; the head adds p after the LayerNorm, x_i = LN(p * c) + p
        #
        #   "learned" h = p * c + L x0, L: x0_dim -> out_dim learned, bias-free (the multiplier p stays the fixed selection)
        #   "pool"    h = p * c + pool(x0), the block of x0 around each selected coordinate, summed / sqrt(k) (no parameters)
        #   "scalar"  h = p * c + alpha * p with one learned scalar alpha per layer (init 1)
        #   "rezero"  h = p + beta * (p * c) with one learned scalar beta per layer (init 0), ReZero on the cross
        self.res_mode = res_mode
        if res_mode == "gain":
            self.alpha = nn.Parameter(torch.ones(out_dim))
        elif res_mode == "comp":
            from .pyramid_baselines import StridedSlice, even_idx
            assert anchor == "copy" and res and not rows, "comp residual is defined for the copy re-read, no masks"
            if out_dim == x0_dim:
                self.proj_res = nn.Identity()
            else:
                stride = x0_dim // out_dim
                ridx = even_idx(x0_dim, out_dim) + anchor_offset + stride // 2
                assert x0_dim % out_dim == 0 and int(ridx.max()) < x0_dim, (x0_dim, out_dim)
                self.proj_res = StridedSlice(ridx)
        elif res_mode == "learned":
            self.proj_res = nn.Linear(x0_dim, out_dim, bias=False)
        elif res_mode == "pool":
            assert x0_dim % out_dim == 0, (x0_dim, out_dim)
            self.proj_res = nn.Identity() if out_dim == x0_dim else GroupPool(x0_dim // out_dim)
        elif res_mode == "scalar":
            self.alpha = nn.Parameter(torch.ones(1))
        elif res_mode == "rezero":
            self.beta = nn.Parameter(torch.zeros(1))
        else:
            assert res_mode in ("same", "post"), res_mode
        # Crossing-form variants ("dcn" is the Table 1 layer
        # h = p * (1 + A x + b), every other form keeps the fixed re-read p and its "+1", no masks):
        #   "relu"    h = p * (1 + ReLU(A x + b))                         QNN T19
        #   "sq"      h = p * (1 + (A x + b)^2)                           QNN T4/T21
        #   "prod"    h = p * (1 + (A x + b) * (B x + b'))                 QNN T16 (GCN-like)
        #   "sig"     h = p * (1 + (A x + b) * sigmoid(G x + g))           GDCN gate
        #   "inaware" h = p * (1 + A x + B p + b)                         QNN T17: the state and the selected input
        #   "split"   h = [p1 * (1 + c1), c2]                            QNN T13: half the units cross, half stay linear
        #   "linexp"  h = [p_even, q] * (1 + A x + b), q evenly spaced coordinates of x   FCN-like: linear + exponential
        #   "two"     h = p * (1 + p2 * (A x + b)), p2 a second selection of x0 (anchor2_idx)
        #   "silu"    h = SiLU(p) * (A x + b) + p
        #   "mix"     h = p * (1 + sum_k g_k(x) U_k V_k x + b), 4 low-rank experts with a softmax gate (DCN-Mix)
        #   "inonly"  h = p * (1 + B x0 + b): the layer crosses the input with itself and ignores the state
        #   "add"     h = (A x + b) + p: no product, the cross enters linearly
        #
        #   "quad"    h = (B x + b') * (A x + b): product of two maps of the state
        #   "glu"     h = (A x + b) * sigmoid(B x + b'): gated linear unit on the state
        #   "self"    h = p * (A p + b): the layer crosses its own multiplier with itself
        self.cross_mode = cross
        self.cross = nn.ModuleList(nn.Linear(out_dim if cross == "self" else in_dim, out_dim)
                                   for _ in range(max(rows, 1))) if cross != "mix" else nn.ModuleList()
        hid = mask_hidden or max(1, int(in_dim * t))  # mask_hidden: explicit hidden width of the mask MLP
        self.masks = nn.ModuleList(
            nn.Sequential(nn.Linear(m_in, hid), nn.ReLU(), nn.Linear(hid, out_dim))
            for _ in range(rows)
        )
        if cross != "dcn":
            assert not rows and res and res_mode == "same", (cross, rows, res, res_mode)
            if cross in ("prod", "sig", "quad", "glu"):
                self.cross2 = nn.Linear(in_dim, out_dim)
            elif cross == "inaware":
                self.cross2 = nn.Linear(out_dim, out_dim, bias=False)
            elif cross == "mix":
                k = 4
                r = max(1, round(in_dim * out_dim / (k * (in_dim + out_dim))))  # the experts cost about one A_i
                self.mix_v = nn.ModuleList(nn.Linear(in_dim, r, bias=False) for _ in range(k))
                self.mix_u = nn.ModuleList(nn.Linear(r, out_dim, bias=False) for _ in range(k))
                self.mix_gate = nn.Linear(in_dim, k)
                self.mix_bias = nn.Parameter(torch.zeros(out_dim))
            elif cross == "linexp":
                from .pyramid_baselines import StridedSlice, even_idx
                self.n_in = (out_dim + 1) // 2
                self.state_sel = StridedSlice(even_idx(in_dim, out_dim - self.n_in))
            elif cross == "two":
                from .pyramid_baselines import StridedSlice
                assert anchor2_idx is not None and len(anchor2_idx) == out_dim, (out_dim,)
                self.proj_x0_2 = StridedSlice(anchor2_idx)
            else:
                assert cross in ("relu", "sq", "split", "silu", "inonly", "add", "self"), cross
        self.apply(compat.init_linear)
        if anchor == "random":
            self.proj_x0.weight.requires_grad_(False)
        for i, m in enumerate(self.masks):
            nn.init.zeros_(m[-1].weight)
            # rows sum to the identity gate at init: each row starts at 1/R
            nn.init.constant_(m[-1].bias, 1.0 / rows)

    def forward(self, x0, x):
        p = self.proj_x0(x0)
        if self.cross_mode != "dcn":
            return self.forward_cross(p, x0, x)
        if self.res_mode != "same":
            assert not self.masks and self.res
            c = self.inter_drop(self.cross[0](x))
            if self.res_mode in ("gain", "scalar"):
                return p * c + self.alpha * p
            if self.res_mode in ("comp", "learned", "pool"):
                return p * c + self.proj_res(x0)
            if self.res_mode == "rezero":
                return p + self.beta * (p * c)
            return p * c  # "post": the head adds p after the LayerNorm
        one = 1 if self.res else 0
        if not self.masks:
            return p * (one + self.inter_drop(self.cross[0](x)))
        src = x0 if self.mask_src == "x0" else x
        return p * (one + self.inter_drop(sum(m(src) * c(x) for m, c in zip(self.masks, self.cross))))


def _forward_cross(self, p, x0, x):
    m = self.cross_mode
    if m == "mix":
        g = torch.softmax(self.mix_gate(x), dim=-1)
        c = sum(g[..., k : k + 1] * u(v(x)) for k, (u, v) in enumerate(zip(self.mix_u, self.mix_v))) + self.mix_bias
        return p * (1 + c)
    if m == "self":
        return p * self.cross[0](p)
    c = self.cross[0](x0 if m == "inonly" else x)
    if m in ("dcn", "inonly"):
        return p * (1 + c)
    if m == "add":
        return c + p
    if m == "relu":
        return p * (1 + torch.relu(c))
    if m == "sq":
        return p * (1 + c * c)
    if m == "prod":
        return p * (1 + c * self.cross2(x))
    if m == "sig":
        return p * (1 + c * torch.sigmoid(self.cross2(x)))
    if m == "quad":
        return c * self.cross2(x)
    if m == "glu":
        return c * torch.sigmoid(self.cross2(x))
    if m == "inaware":
        return p * (1 + c + self.cross2(p))
    if m == "split":
        n = p.shape[-1] // 2
        return torch.cat([p[..., :n] * (1 + c[..., :n]), c[..., n:]], dim=-1)
    if m == "linexp":
        return torch.cat([p[..., 0::2], self.state_sel(x)], dim=-1) * (1 + c)
    if m == "two":
        return p * (1 + self.proj_x0_2(x0) * c)
    if m == "silu":
        return torch.nn.functional.silu(p) * c + p
    raise ValueError(m)


CondLayer.forward_cross = _forward_cross

