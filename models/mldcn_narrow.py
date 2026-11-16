"""ML-DCN with constant/downward V, U and mask maps."""
import torch
from torch import nn
from torch.nn import functional as F
from .embedding import ProjEmbedding
from .layers import StridedSlice, balanced_idx

class Gather(nn.Module):
    def __init__(self, source, target):
        super().__init__()
        assert 1 <= target <= source
        self.register_buffer('idx', torch.arange(target) * source // target)

    def forward(self, x):
        return x.index_select(-1, self.idx)

class NarrowMLDCN(nn.Module):
    def __init__(self, vocab, fields, emb, width, dims, cross_ranks, mask_hidden_widths):
        super().__init__()
        assert len(dims) == 3 and dims[0] == width
        self.embedding = ProjEmbedding(vocab, fields, emb, width)
        prevs = [width] + dims[:-1]
        self.ranks = list(cross_ranks)
        middles = list(mask_hidden_widths)
        assert all(p >= h >= r >= d for p,h,r,d in zip(prevs,middles,self.ranks,dims))
        self.V = nn.ModuleList(nn.Linear(p,r,bias=False) for p,r in zip(prevs,self.ranks))
        self.U = nn.ModuleList(nn.Linear(r,d) for r,d in zip(self.ranks,dims))
        self.mask = nn.ModuleList(nn.Sequential(nn.Linear(p,h),nn.ReLU(),nn.Linear(h,r))
                                  for p,h,r in zip(prevs,middles,self.ranks))
        for branch in self.mask:
            nn.init.zeros_(branch[-1].weight)
            nn.init.ones_(branch[-1].bias)
        self.input_connection = nn.ModuleList(StridedSlice(idx) for idx in balanced_idx(width,dims))
        self.state_connection = nn.ModuleList(Gather(p,d) for p,d in zip(prevs,dims))
        self.norms = nn.ModuleList(nn.LayerNorm(d) for d in dims)
        self.fc = nn.Linear(dims[-1],1)

    def forward(self, dense, cats):
        x = self.embedding(cats).flatten(1)
        h = x
        for v,u,mask,r,s,norm in zip(self.V,self.U,self.mask,self.input_connection,self.state_connection,self.norms):
            h = norm(r(x)*u(v(h)*mask(h)) + s(h))
        return self.fc(h).squeeze(-1)

    def loss(self, cats, labels):
        return F.binary_cross_entropy_with_logits(self(None,cats),labels)
