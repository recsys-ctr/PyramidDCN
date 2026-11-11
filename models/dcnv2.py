"""DCNv2 on the shared projected input, and its pyramidal conversion (Table 2).

Low-rank cross layers next to a parallel DNN; the cross layer follows the FuxiCTR implementation.
    Table 1   rank floor(W / 4), constant widths
    Table 2   original: rank W; pyramidal conversion: cross_widths and ranks narrow by layer, as does the DNN
"""
import torch
import torch.nn as nn

from .base import BaseModel
from .layers import MLP_Block, StridedSlice, even_idx


class DCNv2(BaseModel):
    """DCNv2 (parallel) on the projected input: cross layers x <- x + x0 * U(V x) with rank r (DCN V2 Eq. 2) and a
    parallel DNN.  With cross_widths, layer l writes U: r_l -> cross_widths[l] instead of restoring the width; the
    state and the x0 anchor are cut by strided slices (the pyramidal DCNv2 of Table 2)."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, num_cross_layers=3, rank=None, dnn_widths=(256,) * 5,
                 net_dropout=0.1, batch_norm=True, cross_widths=None, ranks=None):
        super().__init__(vocab_size, num_fields, emb_dim)
        d0 = num_fields * emb_dim
        d = d0
        r = rank or d // 4
        self.d0, self.d, self.r, self.dnn_widths = d0, d, r, list(dnn_widths)
        self.cross_widths = list(cross_widths) if cross_widths is not None else None
        if self.cross_widths is None:
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

    def logits(self, feature_emb):
        x0 = feature_emb.flatten(start_dim=1)
        x = x0
        if self.cross_widths is None:
            for V, U in zip(self.V, self.U):
                x = x + x0 * U(V(x))  # DCN V2 Eq. (2)
        else:
            for V, U, S, P in zip(self.V, self.U, self.skip, self.anchor):
                x = S(x) + P(x0) * U(V(x))
        return self.fc(torch.cat([x, self.parallel_dnn(x0)], dim=-1)).squeeze(-1)
