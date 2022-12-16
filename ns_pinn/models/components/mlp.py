from typing import Callable

import torch
import torch.nn as nn


# TODO: add normalization
class MLP(nn.Module):
    def __init__(self, layer_sizes: list[int], activation_layer: Callable[[], nn.Module], dropout: float = 0.0) -> None:
        super(MLP, self).__init__()
        layers = []
        in_dim = layer_sizes[0]
        for out_dim in layer_sizes[1:-1]:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation_layer())
            if dropout:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, layer_sizes[-1]))
        self.net = nn.Sequential(*layers)
        self.net.apply(self._init_layer)

    def _init_layer(self, layer: nn.Module):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
