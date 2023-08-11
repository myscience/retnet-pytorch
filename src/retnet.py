import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple

from math import ceil

from .utils import default
from .utils import MLP
from .msr import MultiScaleRetention


class RetNet(nn.Module):
    '''
        Retentive Network as introduced in:
        `Retentive Network: A Successor to Transformer for Large Language Models`
        (https://arxiv.org/abs/2307.08621)

        This module represents a potential advancement to the dominant Transformer
        architecture as it combined a parallel and a recurrent formulation of the
        attention mechanism. In particular, the recurrent formulation scales nicely
        with the sequence length (as opposed to the costly Transformer) and early
        tests reported in the paper indicate that inference speed and scaling laws
        compare favorably with the transformer.
    '''

    def __init__(
        self,
        num_layer : int = 6,
        num_heads : int = 8,
        dim_model : int = 512,
        dropout : float = 0.1,
        value_factor : int = 2,
        msr_gate_fn : str = 'gelu',
        mlp_gate_fn : str = 'gelu',
        mlp_mult : int | List[int] = 4,
        mlp_bias : bool = True,
    ) -> None:
        super().__init__()

        self.num_layer = num_layer
        self.num_heads = num_heads
        self.dim_model = dim_model

        self.layers : List[Tuple[MultiScaleRetention, MLP]] = nn.ModuleList(
            [nn.ModuleList(
                (
                    MultiScaleRetention(
                        dim_model = dim_model,
                        num_heads = num_heads,
                        gate_fn = msr_gate_fn,
                        value_factor = value_factor
                    ),
                    MLP(
                        dim_model = dim_model,
                        dim_mult = mlp_mult,
                        gate_fn = mlp_gate_fn,
                        dropout = dropout,
                        bias = mlp_bias
                    )
                )
            ) for _ in range(num_layer)]
        )

    def forward(
        self,
        x : Tensor,
        num_chunk : int | None = None,
        attn_mask : Tensor | str | None = None,
        **kwargs,
    ) -> Tensor:
        '''
            Forward pass of the RetNet. Can use either the parallel implementation
            of the multi-scale retention (a.k.a. attention) mechanism or the
            chunk-recurrent implementation. The parallel implementation is faster
            (for training) but more memory hungry, while chunk-recurrent has lower
            memory consumption and is ideal for long-sequences at inference time.

            Args:
            - x [Tensor]: Input tensor of shape [batch_size, seq_len, dim_model]

            Params:
            - num_chunk [int|None]: Number of chunk to split the input sequence into.
                Use num_chunk=None (no-splitting) to trigger the parallel computation.
                Use num_chunk=-1 (full-chunking) to trigger the fully-recurrent computation.
                    (number of chunks equals sequence length)
                Default: None

            - attn_mask [Tensor|None]: Attention mask to apply to the input sequence.
                Default: None

            Returns:
            - x [Tensor]: Output tensor of shape [batch_size, seq_len, dim_model]
        '''

        bs, seq_len, d_model = x.shape

        if isinstance(attn_mask, str) and attn_mask == 'causal':
            attn_mask = self._get_causal_mask(x, num_chunk)

        if num_chunk: num_chunk = min(num_chunk, seq_len)
        if num_chunk and num_chunk == -1: num_chunk = seq_len
        if num_chunk and num_chunk < 0: raise ValueError('Number of chunks should be positive or equal to -1')

        for msr, mlp in self.layers:
            # These are eq.(9) in the original paper
            # NOTE: Both MSR and MLP do layer normalization on their input
            #       before the attention and MLP computations.
            x = msr(x, num_chunk=num_chunk, attn_mask=attn_mask, **kwargs) + x
            x = mlp(x) + x

        return x

    def _get_causal_mask(self, x : Tensor, num_chunk : int | None) -> Tensor:
        (_, seq_len, _), device = x.shape, x.device

        num_chunk = default(num_chunk, 1)

        attn_dim = ceil(seq_len / num_chunk)
        attn_val = torch.arange(attn_dim, device=device, dtype=float)

        attn_mask = torch.tril(torch.ones(attn_dim, attn_dim, device=device))
        attn_mask = torch.masked_fill(attn_val[:, None] - attn_val[None, :], ~attn_mask.bool(), torch.inf)

        return attn_mask
