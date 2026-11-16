"""Original constant-width ML-DCN (arXiv:2602.09194).

The selected converted model is NarrowMLDCN in mldcn_narrow.py.
"""
import torch.nn as nn


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
