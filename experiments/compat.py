import torch.nn as nn

batch_norm_kwargs = {
    "eps": 1e-3,
    "momentum": 0.01,
}

layer_norm_kwargs = {
    "eps": 1e-3,
}


def init_linear(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
