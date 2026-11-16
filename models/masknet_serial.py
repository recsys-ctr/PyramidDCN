"""Serial MaskNet configurations selected for Tables 1 and 2."""
import math
import torch
from torch import nn
from .base import BaseModel
from .layers import StridedSlice, balanced_idx
from .masknet import MaskBlock

class CrossOnlyMaskNet(BaseModel):
    def __init__(self, vocab, fields, emb, arm):
        super().__init__(vocab, fields, emb)
        W = fields * emb
        self.widths = arm['cross_widths']
        self.converted = arm['condition'] != 'original'
        assert len(self.widths) == 3 and arm['mlp_widths'] == []
        assert not arm['emb_layernorm'] and arm['net_layernorm']
        assert not self.converted
        assert self.widths == [W] * 3
        self.blocks = nn.ModuleList(MaskBlock(W,W,W,'relu',2,0.,True) for _ in range(3))
        self.fc = nn.Linear(self.widths[-1],1)
        self.reset_parameters()

    def logits(self, emb):
        x=emb.flatten(1);h=x
        for i,block in enumerate(self.blocks):
            h=block(x,h)
        return self.fc(h).squeeze(-1)

class NarrowMaskBlock(nn.Module):
    def __init__(self, previous_width, output_width, variant):
        super().__init__()
        assert variant == 'poolmix' and previous_width % output_width == 0
        self.variant = variant
        self.previous_width = previous_width
        self.output_width = output_width
        mask_width = output_width
        self.mask_layer = nn.Sequential(nn.Linear(mask_width, mask_width), nn.ReLU(),
                                        nn.Linear(mask_width, mask_width))
        map_input = output_width
        self.state_map = nn.Linear(map_input, output_width, bias=False)
        self.norm = nn.LayerNorm(output_width)
        self.activation = nn.ReLU()

    def forward(self, selected_input, hidden):
        mask = self.mask_layer(selected_input)
        # Compress adjacent hidden channels before input-conditioned masking.
        ratio = self.previous_width // self.output_width
        pooled = hidden.reshape(*hidden.shape[:-1], self.output_width, ratio).sum(-1) / math.sqrt(ratio)
        h = self.state_map(mask * pooled)
        return self.activation(self.norm(h))

class AlternativeMaskNet(BaseModel):
    def __init__(self, vocab, fields, emb, arm):
        super().__init__(vocab, fields, emb)
        width = fields * emb
        self.widths = arm['cross_widths']
        variant = arm['mask_variant']
        assert self.widths == [width, width // 2, width // 4]
        assert arm['mlp_widths'] == [] and not arm['emb_layernorm'] and arm['net_layernorm']
        previous = [width] + self.widths[:-1]
        mask_widths = self.widths
        self.input_selection = nn.ModuleList(StridedSlice(idx) for idx in balanced_idx(width, mask_widths))
        self.blocks = nn.ModuleList(NarrowMaskBlock(p, d, variant) for p, d in zip(previous, self.widths))
        self.fc = nn.Linear(self.widths[-1], 1)
        self.reset_parameters()

    def logits(self, emb):
        x = emb.flatten(1)
        hidden = x
        for select, block in zip(self.input_selection, self.blocks):
            hidden = block(select(x), hidden)
        return self.fc(hidden).squeeze(-1)
