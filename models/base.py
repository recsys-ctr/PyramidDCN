"""Base class of the baselines that follow the FuxiCTR model interface.

A model owns one embedding table shared by all fields (global ids) and implements logits(feature_emb):
    forward(dense, cats) -> logits FloatTensor[B]   (dense ignored; cats int [B, num_fields])
    loss(cats, y)        -> scalar training loss (BCE unless the model defines its own)
In the experiments train.py replaces the table by the shared projected input (ProjEmbedding).

FuxiCTR conventions reproduced:
  * embedding ~ N(0, 1e-4^2) (FeatureEmbeddingDict default initializer)
  * BaseModel.reset_parameters: every module whose type is exactly nn.Linear / nn.Conv1d gets
    xavier_normal_ weight and zero bias; custom params (FinalMLP w_xy) keep the init done in their own __init__;
    LayerNorm / BatchNorm keep torch defaults.
  * the official output Sigmoid is folded into BCE-with-logits (mathematically identical loss).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
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
