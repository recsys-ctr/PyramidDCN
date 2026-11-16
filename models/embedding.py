"""Input projection shared by the released experiments."""
import torch
from torch import nn

class ProjEmbedding(nn.Module):
    """Shared input stage of every model: one EMB-wide table, flattened (d0), then a bias-free Linear d0 -> W
    (xavier normal, as FuxiCTR initializes Linear layers).  Every backbone sees the same W-wide input, viewed as
    W/8 groups of width 8 (models that only flatten their input are unaffected by the grouping)."""

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
