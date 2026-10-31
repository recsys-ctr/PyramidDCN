"""Hand ports of three more official CTR baselines (FCN, GDCN, FiBiNet++) for the Criteo_x4 harness.

Same API / conventions as x4_baselines.py (which is imported, never modified):
    build_extra(name, vocab_size, num_fields=39, emb_dim=16, **overrides) -> nn.Module
      .embedding            single nn.Embedding(vocab_size, emb_dim * heads) shared by all fields (global ids)
      .forward(dense, cats) -> logits FloatTensor[B]   (dense ignored; cats int [B, num_fields])
      .loss(cats, y)        -> scalar training loss (official loss: Tri-BCE for FCN, BCE otherwise)
      .learned_maps()       -> every learned map / width-changing op in forward order (for the up-projection audit)
      .width_profile()      -> {stage: [representation width after each layer]}

Names / official sources (pinned commits; see OFFICIAL_EXTRA for configs, logs and metrics):
    fcn       FCN_sp (paper's best Criteo row) = salmon1802/FCN@KDD'26 (8eaa864) src/FCNv2.py:
              Exponential2LinearCrossNetwork || Linear2ExponentialCrossNetwork, LCA cross layers, Tri-BCE.
              Tuned config/log: checkpoints/Criteo_x4_10_h5/FCNv2_Criteo_017_30a838c6.log (exp=4, lin=2).
    fcn_p     FCN_p = same branch src/FCNv1.py (ECN || LCN), log FCNv1_Criteo_019_977364ed.log (exp=4, lin=4).
    gdcn      GDCN-P = FuxiCTR v2.2.0 model_zoo/GDCN/src/GDCN.py GDCNP (byte-identical GateCorssLayer to the
              authors' anonctr/GDCN models/GDCN.py).  NO official Criteo_x4 config/log exists (not in
              BARS/RecBench, FuxiCTR yaml only has untuned templates), so the paper's stated setting is used
              (3 gated cross layers, DNN 400-400-400 ReLU, dropout 0.5, BN in DNN as in the authors' MLP).
    gdcn_s    GDCN-S = FuxiCTR GDCNS (renamed GDCN on FuxiCTR main), same paper setting.
    fibinetpp FiBiNet++ = recommendation-algorithm/FiBiNet (af5558b, TensorFlow 1.14) fibinet/run.py
              --version "++" + fibinet/model/fibinet_model.py + components/layers.py.  No logs published;
              the paper's Criteo hyper-parameters are used (reduction ratio 3, group number 2, low-rank 50,
              DNN 3x400 ReLU), except emb_dim (paper 10, harness 16).

Initialisation:
  * fcn / gdcn: FuxiCTR conventions (embedding N(0, 1e-4^2); exact-type nn.Linear -> xavier_normal_, bias 0;
    custom params keep their own init: FCN b ~ U(0,1), gamma=1, beta=0; GDCN b ~ U(0,1)).
  * fibinetpp: the official TF/Keras init (embedding N(0, 0.01^2); W_1/W_2/bilinear W/DNN kernels glorot_normal
    = TF1 truncated normal; final Dense(1) kernel glorot_uniform; all biases 0; BN/LN gamma=1, beta=0).
    Pass init="fuxictr" to use the FuxiCTR convention instead.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .x4_baselines import (  # noqa: F401  (re-exported for check scripts)
    FUXI_EMB_ROWS_X4_001,
    MLP_Block,
    _EMB_PARAMS_16,
    _FuxiCTRModel,
    dense_params,
    multihead_feature_emb,
    serving_macs,
)

_FUXI = "https://github.com/reczoo/FuxiCTR/blob/v2.2.0"
_FCN_SHA = "8eaa8645eed045cea9c1e41eba097627b614f054"  # head of branch KDD'26 (2025-12)
_FCN_K = f"https://github.com/salmon1802/FCN/blob/{_FCN_SHA}"  # == /blob/KDD'26
_GDCN_SHA = "5165518f2bc312274c1d593e62c4ad16a6a7f989"
_GDCN_A = f"https://github.com/anonctr/GDCN/blob/{_GDCN_SHA}"
_FIBI_SHA = "af5558bfe3915c266f233ae0c83770712ea96cbd"
_FIBI = f"https://github.com/recommendation-algorithm/FiBiNet/blob/{_FIBI_SHA}"

# Official tuned configs, parameter counts and metrics (same schema as x4_baselines.OFFICIAL).
# total_params/valid/test are transcribed from the official logs; None = no official log exists.
OFFICIAL_EXTRA = {
    "fcn": dict(
        code=f"{_FCN_K}/src/FCNv2.py",
        log=f"{_FCN_K}/checkpoints/Criteo_x4_10_h5/FCNv2_Criteo_017_30a838c6.log",
        config_file=f"{_FCN_K}/config/model_config.yaml#FCNv2_Criteo (template exp=4, lin=1) + "
                    f"{_FCN_K}/config/FCNv2_tuner_config_Criteo.yaml (grid exp, lin in 1..5)",
        fuxictr="2.0.1", batch_size=8192, embedding_regularizer=1e-5, lr=1e-3, seed=2025,
        config=dict(structure="sp", exp_num_layers=4, lin_num_layers=2, net_dropout=0.1, batch_norm=False,
                    layer_norm=False, num_heads=1),
        total_params=16_924_434,
        valid=dict(auc=0.815818, logloss=0.436275), test=dict(auc=0.816352, logloss=0.435790),
        paper=dict(fcn_sp=dict(auc=0.8163, logloss=0.4357,
                               src="https://arxiv.org/abs/2407.13349v8 Tab.3 FCN_sp (KDD'26); the paper truncates "
                                   "to 4 decimals (log 0.435790/0.816352 -> 0.4357/81.63, cf. ECN 0.436589 -> 0.4365)")),
        alt_log=dict(note="best VALID AUC of the FCNv2 grid (exp=5, lin=5)",
                     log=f"{_FCN_K}/checkpoints/Criteo_x4_10_h5/FCNv2_Criteo_025_747baf17.log",
                     config=dict(exp_num_layers=5, lin_num_layers=5), total_params=18_491_922,
                     valid=dict(auc=0.815861, logloss=0.436210), test=dict(auc=0.816318, logloss=0.435812)),
        notes="paper text says batch 10,000; the official Criteo logs use 8192. FCNv2 logs dated 2025-05-11 "
              "(keys ECN_num_layers/LCN_num_layers) belong to an older, different FCNv2 and are ignored."),
    "fcn_p": dict(
        code=f"{_FCN_K}/src/FCNv1.py",
        log=f"{_FCN_K}/checkpoints/Criteo_x4_10_h5/FCNv1_Criteo_019_977364ed.log",
        config_file=f"{_FCN_K}/config/model_config.yaml#FCNv1_Criteo + FCNv1_tuner_config_Criteo.yaml",
        fuxictr="2.0.1", batch_size=8192, embedding_regularizer=1e-5, lr=1e-3, seed=2024,
        config=dict(structure="p", exp_num_layers=4, lin_num_layers=4, net_dropout=0.1, batch_norm=False,
                    layer_norm=False, num_heads=1),
        total_params=16_140_690,
        valid=dict(auc=0.815565, logloss=0.436539), test=dict(auc=0.816017, logloss=0.436117),
        paper=dict(fcn_p=dict(auc=0.8160, logloss=0.4361, src="https://arxiv.org/abs/2407.13349v8 Tab.3 FCN_p"))),
    "gdcn": dict(
        code=f"{_FUXI}/model_zoo/GDCN/src/GDCN.py#GDCNP",
        authors_code=f"{_GDCN_A}/models/GDCN.py#GDCNP (+ models/basic_layers.py MultiLayerPerceptron, train_base.py)",
        log=None,
        log_status="not found: no GDCN in reczoo/BARS (now reczoo/RecBench) ranking/ctr; FuxiCTR "
                   "model_zoo/GDCN/config/model_config.yaml has only untuned templates (GDCN_default: GDCNP in "
                   "v2.2.0, GDCN(=stacked) on main, emb 32, [1024,512,256], dropout 0, dataset TBD); the authors' "
                   "repo has no logs",
        config_file="paper CIKM'23 Sec.5.1 'Implementation details' + authors' train_base.py defaults",
        framework="pytorch (authors) / fuxictr>=2.0 (FuxiCTR port)",
        batch_size=4096, embedding_regularizer=None, lr=1e-3,
        optimizer_note="authors: Adam(lr=1e-3, weight_decay=1e-6 on ALL params), ReduceLROnPlateau(max, x0.1); "
                       "paper: LR/10 after 3 stagnant epochs, early-stop patience 5",
        config=dict(dnn_hidden_units=[400, 400, 400], dnn_activations="relu", num_cross_layers=3,
                    net_dropout=0.5, batch_norm=True),
        total_params=None, valid=None, test=None,
        official_dense_params=2_912_353,
        official_dense_params_src="authors' GDCNP(field_dims, embed_dim=16, cn_layers=3, mlp_layers=(400,400,400)) "
                                  "instantiated, embedding excluded (no log exists)",
        paper=dict(gdcn_p=dict(auc=0.8161, logloss=0.4360, total_params="20.3M",
                               src="https://doi.org/10.1145/3583780.3615089 Tab.4 (metrics), Tab.6 Fixed (params)"),
                   gdcn_s=dict(auc=0.8158, logloss=0.4364, src="Tab.3")),
        dataset_note="paper/authors use their own random 8:1:1 Criteo split, min count 10, 1,086K features "
                     "(vs FuxiCTR Criteo_x4_001: 910,747 rows); 20.3M - 1.086M*16 ~= 2.91M dense, consistent."),
    "gdcn_s": dict(
        code=f"{_FUXI}/model_zoo/GDCN/src/GDCN.py#GDCNS",
        authors_code=f"{_GDCN_A}/models/GDCN.py#GDCNS",
        log=None, log_status="not found (see gdcn)", batch_size=4096, embedding_regularizer=None, lr=1e-3,
        config=dict(dnn_hidden_units=[400, 400, 400], dnn_activations="relu", num_cross_layers=3,
                    net_dropout=0.5, batch_norm=True),
        total_params=None, valid=None, test=None, official_dense_params=2_911_729,
        official_dense_params_src="authors' GDCNS instantiated, embedding excluded",
        paper=dict(gdcn_s=dict(auc=0.8158, logloss=0.4364, src="Tab.3"))),
    "fibinetpp": dict(
        code=f"{_FIBI}/fibinet/model/fibinet_model.py (+ fibinet/model/components/layers.py, fibinet/run.py "
             f"--version '++')",
        log=None,
        log_status="not found: the official repo publishes no logs/checkpoints; FiBiNet++ is not in "
                   "BARS/RecBench or FuxiCTR (only FiBiNET v1)",
        config_file=f"{_FIBI}/fibinet/run.py (_update_parameters '++', argparse defaults) + "
                    f"{_FIBI}/config/criteo/config_dense.json",
        framework="tensorflow-gpu 1.14", batch_size=1024, embedding_regularizer=0.0, lr=1e-4,
        config=dict(sparse_embedding_norm="bn", senet_group_num=2, senet_reduction_ratio=3.0, low_rank=50,
                    dnn_hidden_units=[400, 400, 400], dnn_activation="relu", dnn_use_bn=False, dnn_dropout=0.0,
                    init_std=0.01),
        official_emb_dim=10,
        total_params=None, valid=None, test=None, official_dense_params=None,
        official_setting_dense_params_analytic=563_643,
        paper=dict(fibinetpp=dict(auc=0.8110, non_emb_params="0.56M",
                                  src="https://arxiv.org/abs/2209.05016 Tab.1 (AUC only; d=10, batch 1024, "
                                      "lr 1e-4, DNN 3x400, reduction ratio 3, group 2, low rank 50)")),
        dataset_note="official Criteo: 13 continuous fields (DenseEmbeddingLayer value*vector + per-field LN) + "
                     "26 categorical (BN), own 10-fold split (parts 0-7/8/9, seed 2018). Harness: all 39 fields are "
                     "FuxiCTR-bucketised categoricals, so all 39 go through the categorical (BN) path."),
}
for _v in OFFICIAL_EXTRA.values():
    if _v.get("total_params") is not None:
        _v["official_dense_params"] = _v["total_params"] - _EMB_PARAMS_16


# --------------------------------------------------------------------------- helpers for the audit
def _lin(name, m, calls=1, note=""):
    return dict(name=name, kind="linear", in_dim=m.in_features, out_dim=m.out_features, calls=calls,
                params=sum(p.numel() for p in m.parameters(recurse=False)), note=note)


def _elem(name, kind, dim, params, note=""):
    return dict(name=name, kind=kind, in_dim=dim, out_dim=dim, calls=1, params=params, note=note)


def _op(name, in_dim, out_dim, note=""):
    return dict(name=name, kind="op", in_dim=in_dim, out_dim=out_dim, calls=1, params=0, note=note)


def _mlp_maps(prefix, mlp: MLP_Block):
    out, mods = [], list(mlp.mlp)
    for i, m in enumerate(mods):
        if isinstance(m, nn.Linear):
            out.append(_lin(f"{prefix}.mlp.{i}", m))
        elif isinstance(m, nn.BatchNorm1d):
            out.append(_elem(f"{prefix}.mlp.{i}", "affine(BN)", m.num_features,
                             sum(p.numel() for p in m.parameters())))
    return out


def _tri_bce(logit_a, logit_b, y):
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


# --------------------------------------------------------------------------- 1. FCN (KDD'26)
class LCACrossNetwork(nn.Module):
    """One class for all four official FCN@KDD'26 towers (FCNv1.ExponentialCrossNetwork / LinearCrossNetwork,
    FCNv2.Exponential2LinearCrossNetwork / Linear2ExponentialCrossNetwork); `anchors[i]` picks layer i's anchor.
    Parameters are created in the official order with the official names (w, b, gamma, beta, layer_norm,
    batch_norm, dropout, fc), so official state_dicts load 1:1.  Per layer (official _cross_layer):
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

    def learned_maps(self, prefix):
        d = self.fc.in_features
        out = []
        for i, a in enumerate(self.anchors):
            kind = "ECN(anchor=x_l)" if a == "self" else "LCN(anchor=x_1)"
            out.append(_lin(f"{prefix}.w.{i}", self.w[i], note=f"layer {i + 1} {kind}: aggregation H1 = W x"))
            if len(self.layer_norm) > i:
                out.append(_elem(f"{prefix}.layer_norm.{i}", "affine(LN)", d // 2, 2 * (d // 2), "H2 = LN(H1)"))
            else:
                out.append(_elem(f"{prefix}.gamma/beta.{i}", "affine", d // 2, 2 * (d // 2), "H2 = gamma*H1 + beta"))
            out.append(_op(f"{prefix}.concat.{i}", d // 2, d,
                           "H = [H1, H2]: width restored by duplicating H1 (non-learned, no matmul)"))
            if len(self.batch_norm) > i:
                out.append(_elem(f"{prefix}.batch_norm.{i}", "affine(BN)", d, 2 * self.batch_norm[i].num_features,
                                 "BN over heads"))
            out.append(_elem(f"{prefix}.b.{i}", "bias", d, d, "x = anchor * (H + b) + x  (residual, width d)"))
        out.append(_lin(f"{prefix}.fc", self.fc, note="tower logit head"))
        return out


class FCN(_FuxiCTRModel):
    """salmon1802/FCN@KDD'26.  structure='sp' -> src/FCNv2.py (FCN_sp: E2LCN || L2ECN);
    structure='p' -> src/FCNv1.py (FCN_p: ECN || LCN).  Multi-head embedding (table width emb_dim*num_heads,
    per-head input F*emb_dim); served logit = (mean_h tower_a + mean_h tower_b) / 2; trained with Tri-BCE."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, structure="sp", exp_num_layers=4, lin_num_layers=2,
                 net_dropout=0.1, batch_norm=False, layer_norm=False, num_heads=1):
        super().__init__(vocab_size, num_fields, emb_dim, table_dim=emb_dim * num_heads)
        self.num_heads, self.structure = num_heads, structure
        input_dim = num_fields * emb_dim  # feature_map.sum_emb_out_dim()
        kw = dict(net_dropout=net_dropout, batch_norm=batch_norm, layer_norm=layer_norm, num_heads=num_heads)
        E, L = exp_num_layers, lin_num_layers
        if structure == "sp":  # FCNv2: towers named E2LCN / L2ECN in the official code
            self.E2LCN = LCACrossNetwork(input_dim, ["self"] * E + ["x0"] * L, **kw)
            self.L2ECN = LCACrossNetwork(input_dim, ["x0"] * L + ["self"] * E, **kw)
            self._towers = ("E2LCN", "L2ECN")
        elif structure == "p":  # FCNv1: ECN / LCN
            self.ECN = LCACrossNetwork(input_dim, ["self"] * E, **kw)
            self.LCN = LCACrossNetwork(input_dim, ["x0"] * L, **kw)
            self._towers = ("ECN", "LCN")
        else:
            raise ValueError(structure)
        self.reset_parameters()

    def branch_logits(self, cats):
        x = multihead_feature_emb(self.embed(cats), self.num_heads)  # B x H x F*D
        a, b = (getattr(self, t) for t in self._towers)
        return a(x).mean(dim=1).squeeze(-1), b(x).mean(dim=1).squeeze(-1)

    def forward(self, dense, cats):
        la, lb = self.branch_logits(cats)
        return (la + lb) * 0.5

    def loss(self, cats, y):
        return _tri_bce(*self.branch_logits(cats), y)

    def learned_maps(self):
        out = [_op("multihead_embedding", self.num_fields * self.emb_dim, self.num_fields * self.emb_dim,
                   "reorder [first halves | second halves] of the field vectors (per head)")]
        for t in self._towers:
            out += getattr(self, t).learned_maps(t)
        out.append(_op("mean_fusion", 2, 1, "(logit_a + logit_b) / 2"))
        return out

    def width_profile(self):
        d = self.num_fields * self.emb_dim
        return {t: [d] * (len(getattr(self, t).anchors) + 1) for t in self._towers}


# --------------------------------------------------------------------------- 2. GDCN (CIKM'23)
class GateCrossLayer(nn.Module):
    """FuxiCTR v2.2.0 model_zoo/GDCN/src/GDCN.py `GateCorssLayer` (identical to anonctr/GDCN models/GDCN.py):
    x_{l+1} = x0 * (W_c x_l + b) * sigmoid(W_g x_l) + x_l,  W_c, W_g: d -> d (no bias), b ~ U(0,1)."""

    def __init__(self, input_dim, cn_layers=3):
        super().__init__()
        self.cn_layers = cn_layers
        self.w = nn.ModuleList([nn.Linear(input_dim, input_dim, bias=False) for _ in range(cn_layers)])
        self.wg = nn.ModuleList([nn.Linear(input_dim, input_dim, bias=False) for _ in range(cn_layers)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros((input_dim,))) for _ in range(cn_layers)])
        for i in range(cn_layers):
            nn.init.uniform_(self.b[i].data)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        for i in range(self.cn_layers):
            xw = self.w[i](x)  # Feature Crossing
            xg = self.activation(self.wg[i](x))  # Information Gate
            x = x0 * (xw + self.b[i]) * xg + x
        return x

    def learned_maps(self, prefix):
        out = []
        for i in range(self.cn_layers):
            out.append(_lin(f"{prefix}.w.{i}", self.w[i], note=f"gated cross layer {i + 1}: cross W_c"))
            out.append(_lin(f"{prefix}.wg.{i}", self.wg[i], note=f"gated cross layer {i + 1}: gate W_g (sigmoid)"))
            out.append(_elem(f"{prefix}.b.{i}", "bias", self.b[i].numel(), self.b[i].numel(),
                             "x = x0*(W_c x + b)*gate + x  (residual)"))
        return out


class GDCNP(_FuxiCTRModel):
    """FuxiCTR v2.2.0 model_zoo/GDCN GDCNP (parallel): logit = fc([GCN(e), DNN(e)])."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, dnn_hidden_units=(), dnn_activations="ReLU",
                 num_cross_layers=3, net_dropout=0.0, batch_norm=False):
        super().__init__(vocab_size, num_fields, emb_dim)
        input_dim = num_fields * emb_dim
        dnn_hidden_units = list(dnn_hidden_units)
        self.dnn = MLP_Block(input_dim, dnn_hidden_units, dnn_activations, None, None, net_dropout, batch_norm) \
            if dnn_hidden_units else None  # in case of only crossing net used
        self.cross_net = GateCrossLayer(input_dim, num_cross_layers)
        self.fc = nn.Linear(dnn_hidden_units[-1] + input_dim, 1)  # official: crashes without a DNN, kept as is
        self.reset_parameters()

    def logits(self, feature_emb):
        feature_emb = feature_emb.flatten(start_dim=1)  # FeatureEmbedding(flatten_emb=True)
        cross_cn = self.cross_net(feature_emb)
        cross_mlp = self.dnn(feature_emb)
        return self.fc(torch.cat([cross_cn, cross_mlp], dim=1)).squeeze(-1)

    def learned_maps(self):
        d = self.num_fields * self.emb_dim
        return (self.cross_net.learned_maps("cross_net") + _mlp_maps("dnn", self.dnn)
                + [_op("concat", self.fc.in_features, self.fc.in_features,
                       f"[GCN out ({d}), DNN out ({self.fc.in_features - d})]"),
                   _lin("fc", self.fc, note="logit head")])

    def width_profile(self):
        d = self.num_fields * self.emb_dim
        mlp_w = [d] + [m.out_features for m in self.dnn.mlp if isinstance(m, nn.Linear)]
        return {"cross_net": [d] * (self.cross_net.cn_layers + 1), "dnn": mlp_w}


class GDCNS(_FuxiCTRModel):
    """FuxiCTR v2.2.0 model_zoo/GDCN GDCNS (stacked; class GDCN on FuxiCTR main): logit = DNN(GCN(e))."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, dnn_hidden_units=(), dnn_activations="ReLU",
                 num_cross_layers=3, net_dropout=0.0, batch_norm=False):
        super().__init__(vocab_size, num_fields, emb_dim)
        input_dim = num_fields * emb_dim
        dnn_hidden_units = list(dnn_hidden_units)
        self.dnn = MLP_Block(input_dim, dnn_hidden_units, dnn_activations, 1, None, net_dropout, batch_norm) \
            if dnn_hidden_units else None
        self.cross_net = GateCrossLayer(input_dim, num_cross_layers)
        self.reset_parameters()

    def logits(self, feature_emb):
        return self.dnn(self.cross_net(feature_emb.flatten(start_dim=1))).squeeze(-1)

    def learned_maps(self):
        maps = self.cross_net.learned_maps("cross_net") + _mlp_maps("dnn", self.dnn)
        maps[-1]["note"] = "logit head"
        return maps

    def width_profile(self):
        d = self.num_fields * self.emb_dim
        mlp_w = [d] + [m.out_features for m in self.dnn.mlp if isinstance(m, nn.Linear)]
        return {"cross_net": [d] * (self.cross_net.cn_layers + 1), "dnn": mlp_w}


# --------------------------------------------------------------------------- 3. FiBiNet++ (CIKM'23)
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

    def learned_maps(self, prefix):
        return [_op(f"{prefix}.squeeze", self.F * self.D, self.W_1.in_features,
                    f"group mean+max: {self.g} groups x 2 stats per field (non-learned)"),
                _lin(f"{prefix}.W_1", self.W_1, note="excitation FC1 (+ReLU)"),
                _lin(f"{prefix}.W_2", self.W_2, note="excitation FC2 -> bit-wise weights A (identity act)"),
                _op(f"{prefix}.reweight", self.F * self.D, self.F * self.D, "V*A + V (re-weight + skip)"),
                _elem(f"{prefix}.reweight_norm", "affine(LN)", self.D, 2 * self.D, "LN over D, shared by fields")]


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

    def learned_maps(self, prefix):
        F_, D = self.F, self.D
        npair = F_ * (F_ - 1) // 2
        return [dict(name=f"{prefix}.W", kind="bilinear", in_dim=D, out_dim=D, calls=F_, params=D * D,
                     note="shared W applied to every field: v_i W"),
                _op(f"{prefix}.pairwise_ip", F_ * D, npair,
                    f"(v_i W) . v_j for i<j: {F_}x{D} -> {npair} scalars (quadratic, no extra params; F*F*D MACs)"),
                _lin(f"{prefix}.low_rank", self.low_rank, note="'Low Rank Layer' (identity act)")]


class FiBiNetPP(_FuxiCTRModel):
    """recommendation-algorithm/FiBiNet@af5558b FiBiNetModel with run.py '--version ++' settings.
    e -> BN (categorical fields) -> {Bi-Linear+ (all_ip + low rank 50), SENet+} -> concat [H_LRL, V_SENet+]
      -> DNN 400-400-400 ReLU -> Dense(1, no bias) + global bias -> sigmoid.  Plain BCE."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, sparse_embedding_norm="bn", senet_group_num=2,
                 senet_reduction_ratio=3.0, low_rank=50, dnn_hidden_units=(400, 400, 400), dnn_activation="relu",
                 dnn_use_bn=False, dnn_dropout=0.0, init_std=0.01, init="official"):
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
        if init == "official":
            self._official_init(init_std)
        elif init == "fuxictr":
            self.reset_parameters()
            nn.init.xavier_normal_(self.bilinear.W)
        else:
            raise ValueError(init)

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

    def learned_maps(self):
        d = self.num_fields * self.emb_dim
        out = []
        if self.emb_norm is not None:
            out.append(_elem("emb_norm", "affine(BN)", self.emb_dim, 2 * self.emb_dim, "BN over D, shared by fields"))
        out += self.bilinear.learned_maps("bilinear") + self.senet.learned_maps("senet")
        cat = self.bilinear.low_rank.out_features + d
        out.append(_op("concat", cat, cat, f"[H_LRL ({self.bilinear.low_rank.out_features}), V_SENet+ ({d})]"))
        out += _mlp_maps("dnn", self.dnn)
        out.append(_lin("out", self.out, note="logit head"))
        out.append(_elem("global_bias", "bias", 1, 1, "PredictionLayer bias"))
        return out

    def width_profile(self):
        d = self.num_fields * self.emb_dim
        npair = self.num_fields * (self.num_fields - 1) // 2
        return {"bilinear": [d, npair, self.bilinear.low_rank.out_features],
                "senet_excitation": [self.senet.W_1.in_features, self.senet.W_1.out_features,
                                     self.senet.W_2.out_features],
                "dnn": [self.dnn.mlp[0].in_features] + [m.out_features for m in self.dnn.mlp
                                                        if isinstance(m, nn.Linear)]}


# --------------------------------------------------------------------------- factory
_BUILDERS = {
    "fcn": FCN,
    "fcn_p": FCN,
    "gdcn": GDCNP,
    "gdcn_s": GDCNS,
    "fibinetpp": FiBiNetPP,
}
EXTRA_BASELINES = tuple(_BUILDERS)


def build_extra(name: str, vocab_size: int, num_fields: int = 39, emb_dim: int = 16, **overrides) -> nn.Module:
    """Build `name` with its official Criteo config (OFFICIAL_EXTRA[name]['config']); kwargs override."""
    key = name.lower()
    if key not in _BUILDERS:
        raise KeyError(f"unknown extra baseline {name!r}; choose from {EXTRA_BASELINES}")
    cfg = dict(OFFICIAL_EXTRA[key]["config"], **overrides)
    model = _BUILDERS[key](vocab_size, num_fields, emb_dim, **cfg)
    model.baseline_name = key
    return model
