"""Hand ports of official FuxiCTR / BARS / salmon1802 CTR baselines, Criteo_x4_001 tuned configs.

API (matches x4_runner.py):
    build_baseline(name, vocab_size, num_fields=39, emb_dim=16, **overrides) -> nn.Module
      .embedding            single nn.Embedding(vocab_size, emb_dim) shared by all fields (global ids)
      .forward(dense, cats) -> logits FloatTensor[B]   (dense ignored; cats int [B, num_fields])
      .loss(cats, y)        -> scalar training loss (BCE, or the model's official extra terms)

Names / official sources (configs are copied from the official Criteo_x4 logs, see OFFICIAL below):
    dcnv2     FuxiCTR v2.2.0 model_zoo/DCNv2          + BARS DCNv2_criteo_x4_001 (log 005_c2376d55)
    masknet   FuxiCTR v2.2.0 model_zoo/MaskNet        + BARS MaskNet_criteo_x4_001 (log 018_ccc857cd)
    finalmlp  FuxiCTR v2.2.0 model_zoo/FinalMLP       + BARS FinalMLP_criteo_x4_001 (log 002_53a37ddd)
    ecn       salmon1802/FCN@master src/ECN.py (self-mask ECN, the 2024 "DCNv3" deep-only net whose
              Criteo row 0.4364 / 81.55 is the ECN row of the QNN paper) + ECN_Criteo / layer_4 log
    dcnv3     salmon1802/FCN@master src/DCNv3.py (ECN + LCN, Tri-BCE)  + DCNv3_Criteo / 011 log
    ecn_lca   salmon1802/FCN@KDD'26 src/ECN.py (gamma/beta "LCA" ECN of FCN arXiv v8, 0.4365 / 81.56)
              + ECN_Criteo / ECN_Criteo_004 log
GDCN: official FuxiCTR port exists (model_zoo/GDCN, v2.2.0) but there is no BARS / official Criteo_x4
config or log, so it is intentionally not included.

FuxiCTR conventions reproduced:
  * embedding ~ N(0, 1e-4^2) (FeatureEmbeddingDict default initializer)
  * BaseModel.reset_parameters: every module whose type is exactly nn.Linear / nn.Conv1d gets
    xavier_normal_ weight and zero bias; custom params (FinalMLP w_xy, ECN/LCN b) keep the init done in
    their own __init__ (xavier_normal_ / uniform_(0,1)); LayerNorm / BatchNorm keep torch defaults.
  * the official output Sigmoid is folded into BCE-with-logits (mathematically identical loss).
  * fields are flattened in feature_cols order (I1..I13, C1..C26) = the order of `cats` columns.
Not reproduced here (runner's job): Adam, (lambda/2)||E||^2 embedding L2 (1e-5), grad-norm clip 10,
lr x0.1 on plateau, early stopping, batch size.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

FUXI_EMB_ROWS_X4_001 = 910_747  # sum of FuxiCTR per-field vocab (incl. __PAD__ and __OOV__) on Criteo_x4_001
_EMB_PARAMS_16 = FUXI_EMB_ROWS_X4_001 * 16  # 14,571,952

_BARS = "https://github.com/reczoo/BARS/tree/main/ranking/ctr"
_FUXI = "https://github.com/reczoo/FuxiCTR/blob/v2.2.0"
_FCN_M = "https://github.com/salmon1802/FCN/blob/master"
_FCN_K = "https://github.com/salmon1802/FCN/blob/KDD'26"

# Official tuned configs, parameter counts and metrics, transcribed from the official logs.
OFFICIAL = {
    "dcnv2": dict(
        code=f"{_FUXI}/model_zoo/DCNv2/src/DCNv2.py",
        log=f"{_BARS}/DCNv2/DCNv2_criteo_x4_001/DCNv2_criteo_x4_001_005_c2376d55.log",
        fuxictr="2.2.0", batch_size=10000, embedding_regularizer=1e-5, lr=1e-3,
        config=dict(model_structure="parallel", use_low_rank_mixture=False, low_rank=32, num_experts=4,
                    num_cross_layers=3, parallel_dnn_hidden_units=[1000] * 5,
                    stacked_dnn_hidden_units=[500, 500, 500], dnn_activations="relu",
                    net_dropout=0.1, batch_norm=True),
        total_params=20_382_577,
        valid=dict(auc=0.814037, logloss=0.438087), test=dict(auc=0.814514, logloss=0.437631)),
    "masknet": dict(
        code=f"{_FUXI}/model_zoo/MaskNet/src/MaskNet.py",
        log=f"{_BARS}/MaskNet/MaskNet_criteo_x4_001/MaskNet_criteo_x4_001_018_ccc857cd.log",
        fuxictr="2.2.0", batch_size=10000, embedding_regularizer=1e-5, lr=1e-3,
        config=dict(model_type="ParallelMaskNet", parallel_num_blocks=2, parallel_block_dim=500,
                    reduction_ratio=0.1, dnn_hidden_units=[1000] * 5, dnn_hidden_activations="relu",
                    net_dropout=0.3, emb_layernorm=False, net_layernorm=False),
        total_params=20_358_077,
        valid=dict(auc=0.812951, logloss=0.439143), test=dict(auc=0.813420, logloss=0.438748)),
    "finalmlp": dict(
        code=f"{_FUXI}/model_zoo/FinalMLP/src/FinalMLP.py",
        log=f"{_BARS}/FinalMLP/FinalMLP_criteo_x4_001/FinalMLP_criteo_x4_001_002_53a37ddd.log",
        fuxictr="2.2.0", batch_size=10000, embedding_regularizer=1e-5, lr=1e-3,
        config=dict(mlp1_hidden_units=[1000] * 5, mlp1_hidden_activations="relu", mlp1_dropout=0.2,
                    mlp1_batch_norm=False, mlp2_hidden_units=[500], mlp2_hidden_activations="relu",
                    mlp2_dropout=0.1, mlp2_batch_norm=False, use_fs=False, fs_hidden_units=[1000],
                    fs1_context=[], fs2_context=[], num_heads=50),
        total_params=19_524_954,
        valid=dict(auc=0.814101, logloss=0.437752), test=dict(auc=0.814584, logloss=0.437353)),
    "ecn": dict(
        code=f"{_FCN_M}/src/ECN.py",
        log=f"{_FCN_M}/checkpoints/Criteo_x4_10_h5/ECN_Criteo_layer_4.log",
        config_file=f"{_FCN_M}/config/model_config.yaml#ECN_Criteo",
        fuxictr="2.0.1", batch_size=4096, embedding_regularizer=1e-5, lr=1e-3,
        config=dict(num_cross_layers=4, net_dropout=0.1, layer_norm=True, batch_norm=False, num_heads=1),
        total_params=15_356_321,
        valid=dict(auc=0.815104, logloss=0.436875), test=dict(auc=0.815572, logloss=0.436456),
        paper=dict(qnn_table_ecn=dict(auc=0.8155, logloss=0.4364, src="https://arxiv.org/abs/2505.17999 Tab.3"),
                   dcnv3_v3_table_dcnv3=dict(auc=0.8155, logloss=0.4364,
                                             src="https://arxiv.org/abs/2407.13349v3 Tab.4"))),
    "dcnv3": dict(
        code=f"{_FCN_M}/src/DCNv3.py",
        log=f"{_FCN_M}/checkpoints/Criteo_x4_10_h5/DCNv3_Criteo_011_d8b8c992.log",
        config_file=f"{_FCN_M}/config/model_config.yaml#DCNv3_Criteo",
        fuxictr="2.0.1", batch_size=4096, embedding_regularizer=1e-5, lr=1e-3,
        config=dict(num_deep_cross_layers=3, num_shallow_cross_layers=4, deep_net_dropout=0.1,
                    shallow_net_dropout=0.2, layer_norm=True, batch_norm=False, num_heads=1),
        total_params=15_944_754,
        valid=dict(auc=0.815693, logloss=0.436281), test=dict(auc=0.816216, logloss=0.435816),
        alt_log=dict(log=f"{_FCN_M}/checkpoints/Criteo_x4_10_h5/DCNv3_Criteo_047_43d9ad18.log",
                     config=dict(num_deep_cross_layers=4, num_shallow_cross_layers=5),
                     total_params=16_336_626, valid=dict(auc=0.815799, logloss=0.436173),
                     test=dict(auc=0.816197, logloss=0.435823))),
    "ecn_lca": dict(
        code=f"{_FCN_K}/src/ECN.py",
        log=f"{_FCN_K}/checkpoints/Criteo_x4_10_h5/ECN_Criteo_004_eb94d575.log",
        config_file=f"{_FCN_K}/config/model_config.yaml#ECN_Criteo",
        fuxictr="2.0.1", batch_size=8192, embedding_regularizer=1e-5, lr=1e-3,
        config=dict(num_cross_layers=4, net_dropout=0.1, layer_norm=False, batch_norm=False, num_heads=1),
        total_params=15_356_321,
        valid=dict(auc=0.815112, logloss=0.437056), test=dict(auc=0.815620, logloss=0.436589),
        paper=dict(fcn_v8_table_ecn=dict(auc=0.8156, logloss=0.4365, src="https://arxiv.org/abs/2407.13349v8 Tab.3"))),
}
for _v in OFFICIAL.values():
    _v["official_dense_params"] = _v["total_params"] - _EMB_PARAMS_16


# --------------------------------------------------------------------------- shared FuxiCTR pieces
def get_activation(name):
    """fuxictr.pytorch.torch_utils.get_activation (string branch)."""
    if not isinstance(name, str):
        return name
    n = name.lower()
    if n == "relu":
        return nn.ReLU()
    if n == "sigmoid":
        return nn.Sigmoid()
    if n == "tanh":
        return nn.Tanh()
    if n == "softmax":
        return nn.Softmax(dim=-1)
    return getattr(nn, name)()


class MLP_Block(nn.Module):
    """fuxictr/pytorch/layers/blocks/mlp_block.py @v2.2.0 (bn_only_once=False):
    [Linear -> (BatchNorm1d) -> act -> (Dropout)] x len(hidden_units) -> (Linear out) -> (out act)."""

    def __init__(self, input_dim, hidden_units=(), hidden_activations="ReLU", output_dim=None,
                 output_activation=None, dropout_rates=0.0, batch_norm=False, use_bias=True):
        super().__init__()
        hidden_units = list(hidden_units)
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        dims = [input_dim] + hidden_units
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=use_bias))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            if hidden_activations[i]:
                layers.append(get_activation(hidden_activations[i]))
            if dropout_rates[i] > 0:
                layers.append(nn.Dropout(p=dropout_rates[i]))
        if output_dim is not None:
            layers.append(nn.Linear(dims[-1], output_dim, bias=use_bias))
        if output_activation is not None:
            layers.append(get_activation(output_activation))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class _FuxiCTRModel(nn.Module):
    """Shared shell: one global embedding table + FuxiCTR BaseModel init + BCE loss."""

    def __init__(self, vocab_size, num_fields, emb_dim, table_dim=None):
        super().__init__()
        self.num_fields, self.emb_dim = num_fields, emb_dim
        self.embedding = nn.Embedding(vocab_size, table_dim or emb_dim)
        nn.init.normal_(self.embedding.weight, std=1e-4)  # FeatureEmbeddingDict default initializer

    def reset_parameters(self):
        """fuxictr BaseModel.reset_parameters: exact-type nn.Linear/nn.Conv1d -> xavier_normal_, bias 0."""
        for m in self.modules():
            if type(m) in (nn.Linear, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def embed(self, cats):  # B x F x D
        if cats.dtype != torch.long:
            cats = cats.long()
        return self.embedding(cats)

    def logits(self, feature_emb):  # B x F x D -> B
        raise NotImplementedError

    def forward(self, dense, cats):
        return self.logits(self.embed(cats))

    def loss(self, cats, y):
        return F.binary_cross_entropy_with_logits(self(None, cats), y.float().view(-1))


# --------------------------------------------------------------------------- 1. DCNv2
class CrossNetV2(nn.Module):
    """fuxictr/pytorch/layers/interactions/cross_net.py @v2.2.0: x_{i+1} = x_i + x_0 * (W_i x_i + b_i)."""

    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.cross_layers = nn.ModuleList(nn.Linear(input_dim, input_dim) for _ in range(num_layers))

    def forward(self, X_0):
        X_i = X_0
        for i in range(self.num_layers):
            X_i = X_i + X_0 * self.cross_layers[i](X_i)
        return X_i


class DCNv2(_FuxiCTRModel):
    """model_zoo/DCNv2/src/DCNv2.py @FuxiCTR v2.2.0."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, model_structure="parallel",
                 use_low_rank_mixture=False, low_rank=32, num_experts=4, stacked_dnn_hidden_units=(),
                 parallel_dnn_hidden_units=(), dnn_activations="ReLU", num_cross_layers=3, net_dropout=0.0,
                 batch_norm=False):
        super().__init__(vocab_size, num_fields, emb_dim)
        input_dim = num_fields * emb_dim
        if use_low_rank_mixture:
            raise NotImplementedError("CrossNetMix not ported (official Criteo_x4 config uses full-rank CrossNetV2)")
        self.crossnet = CrossNetV2(input_dim, num_cross_layers)
        assert model_structure in ["crossnet_only", "stacked", "parallel", "stacked_parallel"]
        self.model_structure = model_structure
        stacked, parallel = list(stacked_dnn_hidden_units), list(parallel_dnn_hidden_units)
        if model_structure in ["stacked", "stacked_parallel"]:
            self.stacked_dnn = MLP_Block(input_dim, stacked, dnn_activations, None, None, net_dropout, batch_norm)
            final_dim = stacked[-1]
        if model_structure in ["parallel", "stacked_parallel"]:
            self.parallel_dnn = MLP_Block(input_dim, parallel, dnn_activations, None, None, net_dropout, batch_norm)
            final_dim = input_dim + parallel[-1]
        if model_structure == "stacked_parallel":
            final_dim = stacked[-1] + parallel[-1]
        if model_structure == "crossnet_only":
            final_dim = input_dim
        self.fc = nn.Linear(final_dim, 1)
        self.reset_parameters()

    def logits(self, feature_emb):
        feature_emb = feature_emb.flatten(start_dim=1)
        cross_out = self.crossnet(feature_emb)
        if self.model_structure == "crossnet_only":
            final_out = cross_out
        elif self.model_structure == "stacked":
            final_out = self.stacked_dnn(cross_out)
        elif self.model_structure == "parallel":
            final_out = torch.cat([cross_out, self.parallel_dnn(feature_emb)], dim=-1)
        else:
            final_out = torch.cat([self.stacked_dnn(cross_out), self.parallel_dnn(feature_emb)], dim=-1)
        return self.fc(final_out).squeeze(-1)


# --------------------------------------------------------------------------- 2. MaskNet
class MaskBlock(nn.Module):
    """MaskNet.py MaskBlock: LN?(W_h (mask(V_emb) * V_hidden)) -> act -> dropout,
    mask = W2 ReLU(W1 V_emb) with W1: input_dim -> int(hidden_dim * reduction_ratio)."""

    def __init__(self, input_dim, hidden_dim, output_dim, hidden_activation="ReLU", reduction_ratio=1,
                 dropout_rate=0, layer_norm=True):
        super().__init__()
        self.mask_layer = nn.Sequential(nn.Linear(input_dim, int(hidden_dim * reduction_ratio)),
                                        nn.ReLU(),
                                        nn.Linear(int(hidden_dim * reduction_ratio), hidden_dim))
        hidden_layers = [nn.Linear(hidden_dim, output_dim, bias=False)]
        if layer_norm:
            hidden_layers.append(nn.LayerNorm(output_dim))
        hidden_layers.append(get_activation(hidden_activation))
        if dropout_rate > 0:
            hidden_layers.append(nn.Dropout(p=dropout_rate))
        self.hidden_layer = nn.Sequential(*hidden_layers)

    def forward(self, V_emb, V_hidden):
        return self.hidden_layer(self.mask_layer(V_emb) * V_hidden)


class SerialMaskNet(nn.Module):
    def __init__(self, input_dim, output_dim=None, hidden_units=(), hidden_activations="ReLU",
                 reduction_ratio=1, dropout_rates=0, layer_norm=True):
        super().__init__()
        hidden_units = list(hidden_units)
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        self.hidden_units = [input_dim] + hidden_units
        self.mask_blocks = nn.ModuleList(
            MaskBlock(input_dim, self.hidden_units[i], self.hidden_units[i + 1], hidden_activations[i],
                      reduction_ratio, dropout_rates[i], layer_norm) for i in range(len(self.hidden_units) - 1))
        # official output_activation (Sigmoid) folded into the loss
        self.fc = nn.Linear(self.hidden_units[-1], output_dim) if output_dim is not None else None

    def forward(self, V_emb, V_hidden):
        v_out = V_hidden
        for blk in self.mask_blocks:
            v_out = blk(V_emb, v_out)
        return self.fc(v_out) if self.fc is not None else v_out


class ParallelMaskNet(nn.Module):
    def __init__(self, input_dim, output_dim=None, num_blocks=1, block_dim=64, hidden_units=(),
                 hidden_activations="ReLU", reduction_ratio=1, dropout_rates=0, layer_norm=True):
        super().__init__()
        self.num_blocks = num_blocks
        self.mask_blocks = nn.ModuleList(
            MaskBlock(input_dim, input_dim, block_dim, hidden_activations, reduction_ratio, dropout_rates,
                      layer_norm) for _ in range(num_blocks))
        # official output_activation (Sigmoid) folded into the loss
        self.dnn = MLP_Block(input_dim=block_dim * num_blocks, output_dim=output_dim, hidden_units=hidden_units,
                             hidden_activations=hidden_activations, output_activation=None,
                             dropout_rates=dropout_rates)

    def forward(self, V_emb, V_hidden):
        return self.dnn(torch.cat([blk(V_emb, V_hidden) for blk in self.mask_blocks], dim=-1))


class MaskNet(_FuxiCTRModel):
    """model_zoo/MaskNet/src/MaskNet.py @FuxiCTR v2.2.0."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, dnn_hidden_units=(64, 64, 64),
                 dnn_hidden_activations="ReLU", model_type="SerialMaskNet", parallel_num_blocks=1,
                 parallel_block_dim=64, reduction_ratio=1, net_dropout=0, emb_layernorm=True, net_layernorm=True):
        super().__init__(vocab_size, num_fields, emb_dim)
        input_dim = num_fields * emb_dim
        if model_type == "SerialMaskNet":
            self.mask_net = SerialMaskNet(input_dim, 1, dnn_hidden_units, dnn_hidden_activations, reduction_ratio,
                                          net_dropout, net_layernorm)
        elif model_type == "ParallelMaskNet":
            self.mask_net = ParallelMaskNet(input_dim, 1, parallel_num_blocks, parallel_block_dim,
                                            list(dnn_hidden_units), dnn_hidden_activations, reduction_ratio,
                                            net_dropout, net_layernorm)
        else:
            raise ValueError(model_type)
        self.emb_norm = nn.ModuleList(nn.LayerNorm(emb_dim) for _ in range(num_fields)) if emb_layernorm else None
        self.reset_parameters()

    def logits(self, feature_emb):
        if self.emb_norm is not None:
            feat_list = feature_emb.chunk(self.num_fields, dim=1)
            V_hidden = torch.cat([self.emb_norm[i](f) for i, f in enumerate(feat_list)], dim=1)
        else:
            V_hidden = feature_emb
        return self.mask_net(feature_emb.flatten(start_dim=1), V_hidden.flatten(start_dim=1)).squeeze(-1)


# --------------------------------------------------------------------------- 3. FinalMLP
class FeatureSelection(nn.Module):
    """FinalMLP.py FeatureSelection, context-free branch (fs*_context == []): gate_k = 2*sigmoid(MLP(b_k))."""

    def __init__(self, feature_dim, embedding_dim, fs_hidden_units=()):
        super().__init__()
        self.fs1_ctx_bias = nn.Parameter(torch.zeros(1, embedding_dim))
        self.fs2_ctx_bias = nn.Parameter(torch.zeros(1, embedding_dim))
        self.fs1_gate = MLP_Block(embedding_dim, list(fs_hidden_units), "ReLU", feature_dim, "Sigmoid", 0.0, False)
        self.fs2_gate = MLP_Block(embedding_dim, list(fs_hidden_units), "ReLU", feature_dim, "Sigmoid", 0.0, False)

    def forward(self, flat_emb):
        gt1 = self.fs1_gate(self.fs1_ctx_bias.repeat(flat_emb.size(0), 1)) * 2
        gt2 = self.fs2_gate(self.fs2_ctx_bias.repeat(flat_emb.size(0), 1)) * 2
        return flat_emb * gt1, flat_emb * gt2


class InteractionAggregation(nn.Module):
    """FinalMLP.py multi-head bilinear fusion: w_x x + w_y y + sum_h x_h^T W_h y_h."""

    def __init__(self, x_dim, y_dim, output_dim=1, num_heads=1):
        super().__init__()
        assert x_dim % num_heads == 0 and y_dim % num_heads == 0
        self.num_heads, self.output_dim = num_heads, output_dim
        self.head_x_dim, self.head_y_dim = x_dim // num_heads, y_dim // num_heads
        self.w_x = nn.Linear(x_dim, output_dim)
        self.w_y = nn.Linear(y_dim, output_dim)
        self.w_xy = nn.Parameter(torch.Tensor(num_heads * self.head_x_dim * self.head_y_dim, output_dim))
        nn.init.xavier_normal_(self.w_xy)

    def forward(self, x, y):
        output = self.w_x(x) + self.w_y(y)
        head_x = x.view(-1, self.num_heads, self.head_x_dim)
        head_y = y.view(-1, self.num_heads, self.head_y_dim)
        xy = torch.matmul(torch.matmul(head_x.unsqueeze(2), self.w_xy.view(self.num_heads, self.head_x_dim, -1))
                          .view(-1, self.num_heads, self.output_dim, self.head_y_dim),
                          head_y.unsqueeze(-1)).squeeze(-1)
        return output + xy.sum(dim=1)


class FinalMLP(_FuxiCTRModel):
    """model_zoo/FinalMLP/src/FinalMLP.py @FuxiCTR v2.2.0 (context-free feature selection only)."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, mlp1_hidden_units=(64, 64, 64),
                 mlp1_hidden_activations="ReLU", mlp1_dropout=0, mlp1_batch_norm=False,
                 mlp2_hidden_units=(64, 64, 64), mlp2_hidden_activations="ReLU", mlp2_dropout=0,
                 mlp2_batch_norm=False, use_fs=True, fs_hidden_units=(64,), fs1_context=(), fs2_context=(),
                 num_heads=1):
        super().__init__(vocab_size, num_fields, emb_dim)
        if len(fs1_context) or len(fs2_context):
            raise NotImplementedError("context-conditioned feature selection not ported (official config: [])")
        feature_dim = emb_dim * num_fields
        mlp1_hidden_units, mlp2_hidden_units = list(mlp1_hidden_units), list(mlp2_hidden_units)
        self.mlp1 = MLP_Block(feature_dim, mlp1_hidden_units, mlp1_hidden_activations, None, None, mlp1_dropout,
                              mlp1_batch_norm)
        self.mlp2 = MLP_Block(feature_dim, mlp2_hidden_units, mlp2_hidden_activations, None, None, mlp2_dropout,
                              mlp2_batch_norm)
        self.use_fs = use_fs
        if use_fs:
            self.fs_module = FeatureSelection(feature_dim, emb_dim, fs_hidden_units)
        self.fusion_module = InteractionAggregation(mlp1_hidden_units[-1], mlp2_hidden_units[-1], 1, num_heads)
        self.reset_parameters()

    def logits(self, feature_emb):
        flat_emb = feature_emb.flatten(start_dim=1)
        feat1, feat2 = self.fs_module(flat_emb) if self.use_fs else (flat_emb, flat_emb)
        return self.fusion_module(self.mlp1(feat1), self.mlp2(feat2)).squeeze(-1)


# --------------------------------------------------------------------------- 4. ECN / DCNv3 (salmon1802/FCN)
def multihead_feature_emb(feature_emb, num_heads):
    """MultiHeadFeatureEmbedding.forward minus the lookup: B x F x (D*H) -> B x H x (F*D).
    Per head: [first half of every field's dims (field-major), then second half of every field's dims]."""
    mh = torch.stack(torch.tensor_split(feature_emb, num_heads, dim=-1), dim=1)  # B x H x F x D
    e1, e2 = torch.tensor_split(mh, 2, dim=-1)  # B x H x F x D/2 each
    return torch.cat([e1.flatten(start_dim=2), e2.flatten(start_dim=2)], dim=-1)


class ExponentialCrossNetwork(nn.Module):
    """FCN@master src/ECN.py (== DCNv3.py ECN, fc named dfc there).  Per layer:
    H = W x (d -> d/2, no bias); mask = ReLU(LN(H)); H = [H, H*mask]; x = x * (H + b) + x; dropout(x)."""

    def __init__(self, input_dim, num_cross_layers=3, layer_norm=True, batch_norm=False, net_dropout=0.1,
                 num_heads=1, anchor="self"):
        super().__init__()
        self.num_cross_layers, self.anchor = num_cross_layers, anchor
        self.layer_norm, self.batch_norm, self.dropout = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.w, self.b = nn.ModuleList(), nn.ParameterList()
        for i in range(num_cross_layers):
            self.w.append(nn.Linear(input_dim, input_dim // 2, bias=False))
            self.b.append(nn.Parameter(torch.empty((input_dim,))))
            if layer_norm:
                self.layer_norm.append(nn.LayerNorm(input_dim // 2))
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(num_heads))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))
            nn.init.uniform_(self.b[i].data)
        self.masker = nn.ReLU()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x0 = x
        for i in range(self.num_cross_layers):
            H = self.w[i](x)
            if len(self.batch_norm) > i:
                H = self.batch_norm[i](H)
            mask = self.masker(self.layer_norm[i](H)) if len(self.layer_norm) > i else self.masker(H)
            H = torch.cat([H, H * mask], dim=-1)
            x = (x if self.anchor == "self" else x0) * (H + self.b[i]) + x
            if len(self.dropout) > i:
                x = self.dropout[i](x)
        return self.fc(x)


def LinearCrossNetwork(input_dim, num_cross_layers=3, layer_norm=True, batch_norm=False, net_dropout=0.1,
                       num_heads=1):
    """FCN@master src/DCNv3.py LinearCrossNetwork: identical to ECN except the anchor is x0 (x = x0*(H+b)+x)."""
    return ExponentialCrossNetwork(input_dim, num_cross_layers, layer_norm, batch_norm, net_dropout, num_heads,
                                   anchor="x0")


class ExponentialCrossNetworkLCA(nn.Module):
    """FCN@KDD'26 src/ECN.py ExponentialCrossNetwork.  Per layer:
    H1 = W x (d -> d/2, no bias); H2 = LN(H1) if layer_norm else gamma*H1 + beta; H = [H1, H2];
    (BN over heads); x = x * (H + b) + x; dropout(x)."""

    def __init__(self, input_dim, num_cross_layers=3, batch_norm=True, layer_norm=False, net_dropout=0.1,
                 num_heads=1):
        super().__init__()
        self.num_cross_layers = num_cross_layers
        self.layer_norm, self.batch_norm, self.dropout = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.w, self.b = nn.ModuleList(), nn.ParameterList()
        self.gamma, self.beta = nn.ParameterList(), nn.ParameterList()
        for i in range(num_cross_layers):
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

    def forward(self, x):
        for i in range(self.num_cross_layers):
            H_1 = self.w[i](x)
            H_2 = self.layer_norm[i](H_1) if len(self.layer_norm) > i else self.gamma[i] * H_1 + self.beta[i]
            H = torch.cat([H_1, H_2], dim=-1)
            if len(self.batch_norm) > i:
                H = self.batch_norm[i](H)
            x = x * (H + self.b[i]) + x
            if len(self.dropout) > i:
                x = self.dropout[i](x)
        return self.fc(x)


class ECN(_FuxiCTRModel):
    """FCN@master src/ECN.py (variant='selfmask') or FCN@KDD'26 src/ECN.py (variant='lca').
    Prediction: mean over heads of the per-head logit (official `.mean(dim=1)`), plain BCE."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, num_cross_layers=3, net_dropout=0.0,
                 layer_norm=True, batch_norm=False, num_heads=1, variant="selfmask"):
        # official MultiHeadFeatureEmbedding table has width emb_dim * num_heads (== emb_dim for num_heads=1)
        super().__init__(vocab_size, num_fields, emb_dim, table_dim=emb_dim * num_heads)
        self.num_heads = num_heads
        input_dim = num_fields * emb_dim  # per-head width = feature_map.sum_emb_out_dim()
        if variant == "selfmask":
            self.ECN = ExponentialCrossNetwork(input_dim, num_cross_layers, layer_norm, batch_norm, net_dropout,
                                               num_heads)
        elif variant == "lca":
            self.ECN = ExponentialCrossNetworkLCA(input_dim, num_cross_layers, batch_norm, layer_norm, net_dropout,
                                                  num_heads)
        else:
            raise ValueError(variant)
        self.reset_parameters()

    def logits(self, feature_emb):
        return self.ECN(multihead_feature_emb(feature_emb, self.num_heads)).mean(dim=1).squeeze(-1)


class DCNv3(_FuxiCTRModel):
    """FCN@master src/DCNv3.py: ECN (deep, self anchor) + LCN (shallow, x0 anchor) on the same multi-head
    embedding; served logit = (mean_h ECN + mean_h LCN) / 2; trained with Tri-BCE (official add_loss)."""

    def __init__(self, vocab_size, num_fields=39, emb_dim=16, num_deep_cross_layers=4, num_shallow_cross_layers=4,
                 deep_net_dropout=0.1, shallow_net_dropout=0.3, layer_norm=True, batch_norm=False, num_heads=1):
        super().__init__(vocab_size, num_fields, emb_dim, table_dim=emb_dim * num_heads)
        self.num_heads = num_heads
        input_dim = num_fields * emb_dim
        self.ECN = ExponentialCrossNetwork(input_dim, num_deep_cross_layers, layer_norm, batch_norm,
                                           deep_net_dropout, num_heads)
        self.LCN = LinearCrossNetwork(input_dim, num_shallow_cross_layers, layer_norm, batch_norm,
                                      shallow_net_dropout, num_heads)
        self.reset_parameters()

    def branch_logits(self, cats):
        x = multihead_feature_emb(self.embed(cats), self.num_heads)
        return self.ECN(x).mean(dim=1).squeeze(-1), self.LCN(x).mean(dim=1).squeeze(-1)

    def forward(self, dense, cats):
        dlogit, slogit = self.branch_logits(cats)
        return (dlogit + slogit) * 0.5

    def loss(self, cats, y):
        """DCNv3.add_loss verbatim (BCE on sigmoid == BCE-with-logits); weights are NOT detached, as in the
        official code: L = L_y + L_d * max(L_d - L_y, 0) + L_s * max(L_s - L_y, 0)."""
        y = y.float().view(-1)
        dlogit, slogit = self.branch_logits(cats)
        loss = F.binary_cross_entropy_with_logits((dlogit + slogit) * 0.5, y)
        loss_d = F.binary_cross_entropy_with_logits(dlogit, y)
        loss_s = F.binary_cross_entropy_with_logits(slogit, y)
        weight_d, weight_s = loss_d - loss, loss_s - loss
        # official uses torch.zeros(1) (-> loss of shape [1]); zeros_like keeps the same value as a 0-d scalar
        weight_d = torch.where(weight_d > 0, weight_d, torch.zeros_like(weight_d))
        weight_s = torch.where(weight_s > 0, weight_s, torch.zeros_like(weight_s))
        return loss + loss_d * weight_d + loss_s * weight_s


# --------------------------------------------------------------------------- factory + accounting
_BUILDERS = {
    "dcnv2": DCNv2,
    "masknet": MaskNet,
    "finalmlp": FinalMLP,
    "ecn": lambda v, f, d, **kw: ECN(v, f, d, variant="selfmask", **kw),
    "ecn_lca": lambda v, f, d, **kw: ECN(v, f, d, variant="lca", **kw),
    "dcnv3": DCNv3,
}
BASELINES = tuple(_BUILDERS)


def build_baseline(name: str, vocab_size: int, num_fields: int = 39, emb_dim: int = 16, **overrides) -> nn.Module:
    """Build `name` with its official tuned Criteo_x4 config (OFFICIAL[name]['config']); kwargs override."""
    key = name.lower()
    if key not in _BUILDERS:
        raise KeyError(f"unknown baseline {name!r}; choose from {BASELINES}")
    cfg = dict(OFFICIAL[key]["config"], **overrides)
    model = _BUILDERS[key](vocab_size, num_fields, emb_dim, **cfg)
    model.baseline_name = key
    return model


def dense_params(model: nn.Module) -> int:
    return sum(p.numel() for n, p in model.named_parameters() if not n.startswith("embedding."))


@torch.no_grad()
def serving_macs(model: nn.Module, num_fields: int = 39) -> int:
    """Per-example MACs of every matmul (Linear / matmul / bmm) in the eval-mode forward, via
    torch.utils.flop_counter (counts mm/addmm/bmm/baddbmm; bias adds, embedding lookup, norms,
    activations and elementwise products contribute 0).  FLOPs = 2 * MACs."""
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    model.eval()
    cats = torch.zeros(1, num_fields, dtype=torch.long, device=model.embedding.weight.device)
    try:
        with FlopCounterMode(display=False) as fc:
            model(None, cats)
    finally:
        model.train(was_training)
    total = fc.get_total_flops()
    assert total % 2 == 0
    return total // 2

