import torch.nn as nn
from torch import Tensor
from typing import Any, List

from itertools import pairwise

def default(var : Any | None, val : Any) -> Any:
    return var if var else val

class MLP(nn.Module):
    '''
        Basic multi-layer perceptron
    '''
    def __init__(
        self,
        dim_model,
        dim_mult : int | List[int] = 4,
        dropout : float = 0.1,
        gate_fn : str = 'gelu',
        bias : bool = True,
    ) -> None:
        super().__init__()

        if isinstance(dim_mult, int): dim_mult = [dim_mult]
        dims = [dim_model, *dim_mult, dim_model]

        match gate_fn:
            case 'silu': GateFn = nn.SiLU
            case 'gelu': GateFn = nn.GELU
            case 'relu': GateFn = nn.ReLU
            case 'selu': GateFn = nn.SELU
            case 'sigmoid': GateFn = nn.Sigmoid
            case _: raise ValueError(f'Unknown gating function {gate_fn}')

        self.net = nn.Sequential(
            nn.LayerNorm(dim_model),
            *[nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(dim_in, dim_out, bias = bias),
                    GateFn() if i < len(dims) - 1 else nn.Identity(),
                ) for i, (dim_in, dim_out) in enumerate(pairwise(dims))
            ]
        )

    def forward(self, x : Tensor) -> Tensor:
        return self.net(x)