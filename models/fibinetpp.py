"""FiBiNet++ and its pyramidal conversion (Table 2).

    FiBiNetPP          hand port of recommendation-algorithm/FiBiNet (af5558b) fibinet/model/fibinet_model.py with
                       the run.py '--version ++' settings (SENet+, Bi-Linear+ with low rank 50, DNN)
    PyramidFiBiNetPP   the same model with DNN widths dnn_hidden_units

Initialisation: the official TF/Keras init (embedding N(0, 0.01^2); W_1/W_2/bilinear W/DNN kernels glorot_normal
= TF1 truncated normal; final Dense(1) kernel glorot_uniform; all biases 0; BN/LN gamma=1, beta=0).  The official
DNN is 3 x 400; the released configurations set its widths (configs/).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel
from .layers import MLP_Block


def _tf_glorot_normal_(w):
    """TF1.14 keras glorot_normal: VarianceScaling(1.0, fan_avg, truncated_normal) -> truncated N(0, s^2) cut at
    +-2s with s = sqrt(2 / (fan_in + fan_out)) / 0.87962566103423978."""
    fan_in, fan_out = w.shape[1], w.shape[0]
    s = math.sqrt(2.0 / (fan_in + fan_out)) / 0.87962566103423978
    return nn.init.trunc_normal_(w, 0.0, s, -2 * s, 2 * s)


class SENetPlus(nn.Module):
    """layers.py SENETLayer with the '++' settings: squeeze 'group_mean_max' (g groups per field, max & mean),
    excitation 'bit' (A in R^{F*D}), activation 'none', skip connection, re-weight norm 'ln' (shared LN over D).
        Z = [mean_g(V), max_g(V)] (2*F*g);  A = W_2 ReLU(W_1 Z) (no biases);  out = LN(V * A + V)."""

    def __init__(self, num_fields, emb_dim, group_num=2, reduction_ratio=3.0):
        super().__init__()
        assert emb_dim % group_num == 0
        self.F, self.D, self.g = num_fields, emb_dim, group_num
        squeeze_dim = 2 * num_fields * group_num  # ('mean' + 'max') * groups
        reduction_size = int(max(1.0, squeeze_dim / reduction_ratio))
        self.W_1 = nn.Linear(squeeze_dim, reduction_size, bias=False)  # official W_1: (input_dim, reduction_size)
        self.W_2 = nn.Linear(reduction_size, num_fields * emb_dim, bias=False)
        self.reweight_norm = nn.LayerNorm(emb_dim, eps=1e-6)  # SimpleLayerNormalization(is_share_params=True)

    def forward(self, V):  # B x F x D
        grp = V.reshape(-1, self.F * self.g, self.D // self.g)
        Z = torch.cat([grp.mean(dim=-1), grp.amax(dim=-1)], dim=-1)  # order: mean then max (official)
        A = self.W_2(F.relu(self.W_1(Z))).view(-1, self.F, self.D)  # 'bit' excitation
        return self.reweight_norm(V * A + V)


class BilinearPlus(nn.Module):
    """layers.py BilinearInteraction(bilinear_type='all_ip', dnn_units=[50], dnn_activation='linear'):
    P = (V W) V^T (one shared D x D W), keep the strict upper triangle (i<j, row-major) -> F(F-1)/2 scalars,
    then the 'Low Rank Layer' = DNNLayer([50], linear) = Linear(F(F-1)/2 -> 50) with bias."""

    def __init__(self, num_fields, emb_dim, low_rank=50):
        super().__init__()
        self.F, self.D = num_fields, emb_dim
        self.W = nn.Parameter(torch.empty(emb_dim, emb_dim))  # used as tensordot(v, W, axes=(-1, 0)) = v @ W
        iu = torch.triu_indices(num_fields, num_fields, offset=1)  # == Utils.get_upper_triangular_indices order
        self.register_buffer("iu_i", iu[0], persistent=False)
        self.register_buffer("iu_j", iu[1], persistent=False)
        self.low_rank = nn.Linear(num_fields * (num_fields - 1) // 2, low_rank)

    def forward(self, V):  # B x F x D
        P = torch.matmul(torch.matmul(V, self.W), V.transpose(1, 2))  # B x F x F
        return self.low_rank(P[:, self.iu_i, self.iu_j])


class FiBiNetPP(BaseModel):
    """recommendation-algorithm/FiBiNet@af5558b FiBiNetModel with run.py '--version ++' settings.
    e -> BN (categorical fields) -> {Bi-Linear+ (all_ip + low rank 50), SENet+} -> concat [H_LRL, V_SENet+]
      -> DNN ReLU -> Dense(1, no bias) + global bias -> sigmoid.  Plain BCE.  Official TF/Keras initialization."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, sparse_embedding_norm="bn", senet_group_num=2,
                 senet_reduction_ratio=3.0, low_rank=50, dnn_hidden_units=(400, 400, 400), dnn_activation="relu",
                 dnn_use_bn=False, dnn_dropout=0.0, init_std=0.01):
        super().__init__(vocab_size, num_fields, emb_dim)
        # keras BatchNormalization(axis=-1) on [B, F, D]: per-channel (D) stats over batch & fields,
        # momentum 0.99 (torch 0.01), epsilon 1e-3
        self.emb_norm = nn.BatchNorm1d(emb_dim, eps=1e-3, momentum=0.01) if sparse_embedding_norm == "bn" else None
        self.bilinear = BilinearPlus(num_fields, emb_dim, low_rank)
        self.senet = SENetPlus(num_fields, emb_dim, senet_group_num, senet_reduction_ratio)
        hidden = list(dnn_hidden_units)
        self.dnn = MLP_Block(low_rank + num_fields * emb_dim, hidden, dnn_activation, None, None, dnn_dropout,
                             dnn_use_bn)
        self.out = nn.Linear(hidden[-1], 1, bias=False)  # Dense(1, use_bias=False)
        self.global_bias = nn.Parameter(torch.zeros(1))  # PredictionLayer(use_bias=True)
        self._official_init(init_std)

    @torch.no_grad()
    def _official_init(self, init_std):
        nn.init.normal_(self.embedding.weight, 0.0, init_std)  # RandomNormal(0, init_std)
        for w in (self.senet.W_1.weight, self.senet.W_2.weight, self.bilinear.W, self.bilinear.low_rank.weight):
            _tf_glorot_normal_(w)
        self.bilinear.low_rank.bias.zero_()
        for m in self.dnn.mlp:
            if isinstance(m, nn.Linear):
                _tf_glorot_normal_(m.weight)
                m.bias.zero_()
        nn.init.xavier_uniform_(self.out.weight)  # keras Dense default kernel_initializer glorot_uniform
        self.global_bias.zero_()

    def logits(self, feature_emb):
        V = feature_emb
        if self.emb_norm is not None:
            V = self.emb_norm(V.transpose(1, 2)).transpose(1, 2)
        h_lrl = self.bilinear(V)  # B x 50
        v_se = self.senet(V).flatten(start_dim=1)  # B x F*D  (senet_bilinear_type='none' -> flatten)
        h = self.dnn(torch.cat([h_lrl, v_se], dim=-1))
        return (self.out(h) + self.global_bias).squeeze(-1)


class PyramidFiBiNetPP(FiBiNetPP):
    """Official FiBiNet++ (SENet+ and Bi-Linear+ unchanged) with DNN widths dnn_hidden_units."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, d=256, dnn_hidden_units=None, **kw):
        super().__init__(vocab_size, num_fields, emb_dim, dnn_hidden_units=list(dnn_hidden_units), **kw)
        self.d = d
