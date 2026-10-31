"""Repo FLOP convention: per-example serving Linear MACs (eval graph), FLOPs = 2 x MAC.

Counts every nn.Linear call via forward hooks (so reused layers count per call);
embedding lookups, activations, LayerNorm, Hadamard and reductions are excluded.
"""
import torch
import torch.nn as nn


@torch.no_grad()
def serving_linear_macs(model: nn.Module, dense: torch.Tensor, cats: torch.Tensor) -> int:
    total = 0

    def hook(module, inputs, output):
        nonlocal total
        rows = inputs[0].numel() // module.in_features
        total += rows * module.in_features * module.out_features

    was_training = model.training
    model.eval()
    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Linear)]
    try:
        model(dense[:1], cats[:1])
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return total
