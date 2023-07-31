import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from einops import rearrange
from einops import einsum

class MultiScaleRetention(nn.Module):
    '''
        Multi-Scale Retention layer as introduced in:
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
        embed_dim : int,
        num_heads : int,
        value_factor : int = 2,
        gate_fn : str = 'silu'
    ) -> None:
        super(MultiScaleRetention, self).__init__()

        assert_msg = f'Embedding dimension ({embed_dim} x {value_factor}) not divisible by the number of attention heads ({num_heads}).'
        assert (embed_dim * value_factor) % num_heads == 0, assert_msg
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.keys_dim = embed_dim // num_heads
        self.head_dim = (embed_dim * value_factor) // num_heads

        self.scaling = self.keys_dim ** -0.5
        self.value_factor = value_factor

        self.decay = torch.log(1 - 2 ** (-5 - torch.arange(num_heads)))

        match gate_fn:
            case 'silu': self.gate_fn = F.silu
            case 'gelu': self.gate_fn = F.gelu
            case 'relu': self.gate_fn = F.relu
            case 'selu': self.gate_fn = F.selu
            case 'sigmoid': self.gate_fn = F.sigmoid
            case _: raise ValueError(f'Unknown gating function {gate_fn}')

        # Build the query, key, value, gate and output projection matrices
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim * value_factor)
        self.g_proj = nn.Linear(embed_dim, embed_dim * value_factor)
        self.o_proj = nn.Linear(embed_dim * value_factor, embed_dim)

        # Scale-invariant group normalization layer
        self.norm = nn.GroupNorm(num_groups=num_heads, num_channels=embed_dim)

    def forward(
        self,
        x : Tensor,
        pos_embed : Tensor | None = None,
        attn_mask : Tensor | None = None,
    ) -> Tensor:
        '''
            Forward pass of the Multi-Scale Retention layer.

            Args: 
        '''

    def _parallel_forward(
        self,
        qry : Tensor,
        key : Tensor,
        val : Tensor,
        attn_mask : Tensor | None = None,
    ) -> Tensor:
        '''
            Parallel implementation of forward pass for the Multi-Scale Retention Layer.
            This is basically equivalent to a standard forward pass for a Transformer network.

            Args:
            - qry [Tensor]: Query tensor of shape [batch_size, num_heads, seq_length, head_dim]
            - key [Tensor]: Key   tensor of shape [batch_size, num_heads, seq_length, head_dim]
            - val [Tensor]: Value tensor of shape [batch_size, num_heads, seq_length, head_dim (*)]

            Params:
            - attn_mask [Tensor]: Optional attention mask to implement causal masking or other
                masking mechanism relevant to the problem structure.

            (*) NOTE: Value head dimension differs from key/query head dimension in the general case
                      by a factor of `value_factor`.

            Returns:
            - attn_out [Tensor]: Output tensor of shape [batch_size, num_heads, seq_length, embed_dim]
        '''

        # Reshape value tensor into multi-head formulation
        val = rearrange(val, 'b n (h d) -> b h n d', h = self.num_heads).contiguous()

        ret_mat = einsum(qry, key, 'b h i d, b h j d -> b h i j')

        if attn_mask is not None:
            ret_mat *= attn_mask
            ret_mat = torch.nan_to_num(ret_mat)

        # Normalize the retention matrix for numerical stability (normalize #3 in the original paper)
        ret_mat /= ret_mat.detach().sum(dim=-1, keepdim=True).abs().clamp(min=1)

        attn_out = einsum(ret_mat, val, 'b h i j, b h j d -> b h i d')
        attn_out = rearrange(attn_out, 'b h n d -> b n (h d)').contiguous()

        return attn_out
    
    def _recurrent_forward(
        self,
        qry : Tensor,
        key : Tensor,
        val : Tensor,
        num_chunk : int = 1,
        attn_mask : Tensor | None = None,
    ) -> Tensor:
        '''
            Recurrent implementation of forward pass for the Multi-Scale Retention Layer.
            This formulation is the main innovation of the Ret-Net architecture as it offer a
            constant-memory inference cost (for input sequence length) which improves on the
            costly quadratic formulation of the default Transformer.

            Args:
            - qry [Tensor]: Query tensor of shape [batch_size, num_heads, seq_length, head_dim]
            - key [Tensor]: Key   tensor of shape [batch_size, num_heads, seq_length, head_dim]
            - val [Tensor]: Value tensor of shape [batch_size, num_heads, seq_length, head_dim (*)]


            (*) NOTE: Value head dimension differs from key/query head dimension in the general case
                      by a factor of `value_factor`.

            Params:
            - num_chunk [int]: Number of chunks to split the input tensor into. This is the main
                parameter to tune for the recurrent formulation.
            - attn_mask [Tensor]: Optional attention mask to implement causal masking or other
                masking mechanism relevant to the problem structure.

            Returns:
            - attn_out [Tensor]: Output tensor of shape [batch_size, seq_length, embed_dim]
        '''

        (bs, num_head, seq_len, emb_dim), device = val.shape, qry.device
        chunk_size = seq_len // num_chunk

        # Chunk all the input tensor into num_chunk chunks and put is as leading dimension for easy of looping
        qry, key, val = map(lambda t : rearrange(t, 'b h (c n) d -> c b h n d', c = num_chunk), (qry, key, val))
        
        # Within each chunk we apply the standard (parallel) forward pass
        inner_attn = einsum(qry, key, 'c b h i d, c b h j d -> c b h i j')
        
        if attn_mask is not None:
            inner_attn *= attn_mask
            inner_attn = torch.nan_to_num(inner_attn)

        # Normalize the within-chunk attention for numerical stability (normalize #3 in the original paper)
        inner_scale = inner_attn.detach().sum(dim=-1, keepdim=True).abs().clamp(min=1)
        inner_attn /= inner_scale

        # Compute the within-chunk attention output
        inner_attn = einsum(inner_attn, val, 'c b h i j, c b h j d -> c b h i d')

        # * Compute the cross-chunk component of attention using recurrent formulation
        block_idxs = 1 + torch.arange(chunk_size)
        cross_decay = torch.exp(self.decay * chunk_size).to(device)
        inner_decay = torch.exp(self.decay * block_idxs).to(device)

        # ! FIXME: Original code normalizes inner_decay using the denominator of the attn_mask
        # !        but I was planning to normalize attn_mask in the forward pass, so this info
        # !        is not available to the recurrent formulation. Maybe we should pass it along?
        # if attn_mask:
        #     scale = attn_mask.sum(dim=-1, keepdim=True).sqrt()
        #     inner_decay /= (scale / scale[])

        KV = einsum(key, val, 'c b h n k, c b h n v -> c b h k v')

        cross_chunk = []
        cross_scale = []

        kv_chunk = torch.zeros_like(KV[0])
        kv_scale = torch.ones((bs, num_head, 1, emb_dim), device = device)

        for kv in KV:
            cross_chunk.append(kv / kv_scale)
            cross_scale.append(kv_scale)

            kv_chunk = kv_chunk * cross_decay + kv
            kv_scale = kv_chunk.detach().abs().sum(dim=-2, keepdim=True).clamp(min=1)

        cross_chunk = torch.stack(cross_chunk, dim=0)
        cross_scale = torch.stack(cross_scale, dim=0)

        # Compute the cross-chunk attention (formula (7) in original paper)
        cross_attn = einsum(qry * inner_decay, cross_chunk, 'c b h n k, c b h k d -> c b h n d')

        # Finally combine within- and cross- attention contributions (with normalizations)
        attn_out = inner_attn / cross_scale + cross_attn / inner_scale

        # Return the proper sequence by recombining the chunks and heads together
        attn_out = rearrange(attn_out, 'c b h n d -> b (c n) (h d)').contiguous()

        return attn_out

        

        


       
