"""Dense parallel DCNv2 selected for Tables 1 and 2.

Three cross layers and three parallel MLP layers; balanced fixed input
selection and previous-state slices match each converted cross width.
"""
import torch
import torch.nn as nn

from .base import BaseModel
from .layers import MLP_Block, StridedSlice, even_idx


from .layers import balanced_idx

class DCNVariant(BaseModel):
    def __init__(self,vocab,fields,embedding_dim,arm):
        super().__init__(vocab,fields,embedding_dim)
        W=fields*embedding_dim
        self.topology=arm['topology'];self.parameterization=arm['parameterization']
        assert self.topology == 'parallel' and self.parameterization == 'full'
        assert arm['ranks'] == []
        widths=arm['cross_widths'];self.V=nn.ModuleList();self.U=nn.ModuleList()
        self.skip=nn.ModuleList();self.anchor=nn.ModuleList()
        indices=balanced_idx(W,widths)
        previous=W
        for i,width in enumerate(widths):
            self.V.append(nn.Identity())
            self.U.append(nn.Linear(previous,width))
            self.skip.append(nn.Identity() if previous==width else StridedSlice(even_idx(previous,width)))
            self.anchor.append(nn.Identity() if W==width else StridedSlice(indices[i]))
            previous=width
        mlp_input=W
        self.dnn=MLP_Block(mlp_input,arm['mlp_widths'],'relu',None,None,0.,True)
        head_input=arm['mlp_widths'][-1]+widths[-1]
        self.fc=nn.Linear(head_input,1)
        self.reset_parameters()

    def logits(self,feature_emb):
        x=feature_emb.flatten(1);h=x
        for v,u,s,a in zip(self.V,self.U,self.skip,self.anchor):
            h=s(h)+a(x)*u(v(h))
        z=torch.cat([h,self.dnn(x)],dim=-1)
        return self.fc(z).squeeze(-1)
