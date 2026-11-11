"""ML-DCN (arXiv:2602.09194) on the shared projected input, and its pyramidal conversion (Table 2)."""
import torch
import torch.nn as nn

from .layers import StridedSlice


class MLDCN(nn.Module):
    """ML-DCN (arXiv:2602.09194, Eq. 8): X_{l+1} = LN(X0 ⊙ (((X_l V_l) ⊙ M_l) U_lᵀ + b_l) + X_l).

    M_l = MLP(X_l): d -> k = t·d -> r (ReLU hidden, linear output), applied in the
    rank-r space.  The paper's Eq. 5 gives a d-wide mask, inconsistent with Eq. 8;
    we use width r.  The mask starts at the identity (W = 0, b = 1).  d is the width of the projected input.
    """

    def __init__(self, d, rank, num_layers, t=0.5):
        super().__init__()
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


class PyramidMLDCN(nn.Module):
    """ML-DCN's own layer with narrowing widths (Table 2):
        x_l = LN(A_l(x0) * (U_l((V_l x_{l-1}) * M_l(x_{l-1})) + b_l) + S_l(x_{l-1})),
    V_l: d_{l-1} -> r_l, M_l: d_{l-1} -> int(t d_{l-1}) -> r_l (identity init), U_l: r_l -> d_l,
    A_l: the input re-read to width d_l (Identity when d_l = d0), S_l: the state shortcut cut to width d_l
    (Identity when equal; a strided slice when narrowing; the input re-read when the first layer widens).
    d0 is the width of the projected input."""

    def __init__(self, d0, dims, ranks, t=0.5):
        super().__init__()
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
                                    StridedSlice(torch.arange(d) * d0 // d) if d > d0 else nn.Linear(d0, d, bias=False)
                                    for d in dims)
        self.skip = []
        for p, d in zip(prevs, dims):
            if d == p:
                self.skip.append(nn.Identity())
            elif d < p:
                self.skip.append(StridedSlice(torch.linspace(0, p - 1, d).round().long()))
            else:  # the first layer widens the input: its residual is the input re-read A_l(x0) (no extra map)
                self.skip.append(None)
        self._skip_mods = nn.ModuleList(m for m in self.skip if m is not None)  # registers the slice buffers

    def forward(self, x0):
        x = x0
        for V, U, mask, norm, A, S in zip(self.V, self.U, self.masks, self.norms, self.anchor, self.skip):
            a = A(x0)
            x = norm(a * U(V(x) * mask(x)) + (a if S is None else S(x)))
        return self.fc(x).squeeze(-1)
