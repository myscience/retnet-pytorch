import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from einops import rearrange
from einops import einsum
from einops import repeat

from typing import Tuple

from .utils import exists
from .utils import default

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
        dim_model : int,
        num_heads : int,
        value_factor : int = 2,
        pre_norm : bool = True,
        gate_fn : str = 'silu',
    ) -> None:
        super(MultiScaleRetention, self).__init__()

        assert_msg = f'Embedding dimension ({dim_model} x {value_factor}) not divisible by the number of attention heads ({num_heads}).'
        assert (dim_model * value_factor) % num_heads == 0, assert_msg
        
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.keys_dim = dim_model // num_heads
        self.vals_dim = (dim_model * value_factor) // num_heads

        self.scaling = self.keys_dim ** -0.5
        self.value_factor = value_factor

        # Parameters used for positional encodings
        angle = 1.0 / (10000 ** torch.linspace(0, 1, dim_model // num_heads // 2))
        angle = repeat(angle, 'n -> n 2')
        angle = rearrange(angle, 'n r -> 1 (n r)')

        decay = torch.log(1 - 2 ** (-5 - torch.arange(num_heads)))

        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)

        match gate_fn:
            case 'silu': self.gate_fn = F.silu
            case 'gelu': self.gate_fn = F.gelu
            case 'relu': self.gate_fn = F.relu
            case 'selu': self.gate_fn = F.selu
            case 'sigmoid': self.gate_fn = F.sigmoid
            case _: raise ValueError(f'Unknown gating function {gate_fn}')

        # Build the query, key, value, gate and output projection matrices
        self.q_proj = nn.Linear(dim_model, dim_model)
        self.k_proj = nn.Linear(dim_model, dim_model)
        self.v_proj = nn.Linear(dim_model, dim_model * value_factor)
        self.g_proj = nn.Linear(dim_model, dim_model * value_factor)
        self.o_proj = nn.Linear(dim_model * value_factor, dim_model)

        # Scale-invariant layer and group normalization layer
        self.pre_norm = nn.LayerNorm(dim_model) if pre_norm else nn.Identity()
        self.norm = nn.GroupNorm(num_groups=num_heads, num_channels=dim_model * value_factor)

    def forward(
        self,
        x : Tensor,
        num_chunk : None | int = None,
        attn_mask : None | Tensor = None,
        pos_embed : None | Tuple[Tensor, Tensor] = None,
    ) -> Tensor:
        '''
            Forward pass of the Multi-Scale Retention layer.

            Args:
            - x [Tensor]: Input tensor of shape [batch_size, seq_length, dim_model]

            Params:
            - pos_embed [Tensor]: Optional positional embedding tensor of shape [seq_length, dim_model]
            - attn_mask [Tensor]: Optional attention mask to implement causal masking or other
                masking mechanism relevant to the problem structure.

            Returns:
            - attn_out [Tensor]: Output tensor of shape [batch_size, seq_length, dim_model]
        '''
        bs, seq_len, d_model = x.shape

        x = self.pre_norm(x)
        gate = self.g_proj(x)

        # If num_chunk is provided, pad input sequence with zeros such that it nicely
        # divides into `num_chunk` chunks.
        if num_chunk and seq_len % num_chunk > 0:
            chk_len = seq_len // num_chunk
            pad_len = num_chunk - (seq_len % chk_len)
            x = F.pad(x, (0, 0, 0, pad_len))

        pos_embed = default(pos_embed, self._get_embed(x))

        # Project input tensor to get the query, key and value
        qry : Tensor = self.q_proj(x)
        key : Tensor = self.k_proj(x)
        val : Tensor = self.v_proj(x)

        # Standard query-key normalization, multiply here before passing to specific forward implementations
        key *= self.scaling

        # Prepare query-key-value to have the appropriate multi-head tensor shape
        qry = rearrange(qry, 'b n (h d) -> b h n d', h = self.num_heads).contiguous()
        key = rearrange(key, 'b n (h d) -> b h n d', h = self.num_heads).contiguous()
        val = rearrange(val, 'b n (h d) -> b h n d', h = self.num_heads).contiguous()

        # Add positional embedding to both key and query vectors
        qry = self._rot_embed(qry, pos_embed)
        key = self._rot_embed(key, pos_embed)

        if exists(attn_mask):
            # Produce different decayed version of the attention mask for each head
            attn_mask = torch.exp(attn_mask * rearrange(self.decay, 'h -> h 1 1'))
            attn_mask = torch.nan_to_num(attn_mask)

            # Normalize the attention mask
            self.attn_scale = attn_mask.sum(dim=-1, keepdim=True).sqrt()
            attn_mask /= self.attn_scale

        if num_chunk:
            output = self._recurrent_forward(
                qry, key, val, num_chunk=num_chunk, attn_mask=attn_mask 
            )
        else:
            output = self._parallel_forward(
                qry, key, val, attn_mask=attn_mask
            )

        # Restore the original input shape (undo padding)
        output = output[..., :seq_len, :]

        # Apply group normalization and non-linear gated connection
        output = self.norm(output)
        output = self.gate_fn(gate) * output

        # Return output projection of computed attention (retention)
        return self.o_proj(output)

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
            - attn_out [Tensor]: Output tensor of shape [batch_size, num_heads, seq_length, dim_model]
        '''

        ret_mat = einsum(qry, key, 'b h i d, b h j d -> b h i j')

        if exists(attn_mask): ret_mat *= attn_mask

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
            - attn_out [Tensor]: Output tensor of shape [batch_size, seq_length, dim_model]
        '''

        (bs, num_head, seq_len, emb_dim), device = val.shape, qry.device
        chunk_size = seq_len // num_chunk

        # * Preliminary work: compute some normalization constants
        block_idxs = 1 + torch.arange(chunk_size)

        cross_decay = torch.exp(self.decay * chunk_size).to(device)
        inner_decay = torch.exp(torch.outer(self.decay, block_idxs)).to(device)
        
        # Cross-decay is applied to each attention head (hence multi-scale), axes have semantics
        # [chunk, batch, head, seq_length, d_model], so we relay on implicit broadcasting for the [chunk, batch]
        # dimensions, but we should add two singleton explicit dimensions for the [seq_len, d_model] axes.
        # Inner-decay is applied to each head and chunk, as for cross-decay we relay on implicit broadcasting
        # for the [chunk, batch] dimensions, but we should add an explicit singleton dimension for the [d_model]
        cross_decay = rearrange(cross_decay, 'h -> h 1 1')
        inner_decay = rearrange(inner_decay, 'h n -> h n 1')

        if hasattr(self, 'attn_scale'):
            # Normalize the inner decay based on the attention mask
            inner_decay /= (self.attn_scale / self.attn_scale[:, -1, None])

        # * Now we move on to the actual attention computation
        # Chunk all the input tensor into num_chunk chunks and put is as leading dimension for easy of looping
        qry, key, val = map(lambda t : rearrange(t, 'b h (c n) d -> c b h n d', c = num_chunk), (qry, key, val))
        
        # Within each chunk we apply the standard (parallel) forward pass
        inner_attn = einsum(qry, key, 'c b h i d, c b h j d -> c b h i j')
        
        if exists(attn_mask): inner_attn *= attn_mask

        # Normalize the within-chunk attention for numerical stability (normalize #3 in the original paper)
        inner_scale = inner_attn.detach().sum(dim=-1, keepdim=True).abs().clamp(min=1)
        inner_attn /= inner_scale

        # Compute the within-chunk attention output (left-hand side of eq.(7))
        inner_attn = einsum(inner_attn, val, 'c b h i j, c b h j d -> c b h i d')

        # Now compute the cross-chunk attention output (right-hand side of eq.(7)),
        # we start by computing the KV term (that forms the retention R) for all the
        # chunks in a single pass and then iteratively add the previous chunk's
        # retention to the current one.
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
    
    def _get_embed(self, x : Tensor) -> Tuple[Tensor, Tensor]:
        '''
            Compute rotational embeddings based on input sequence length.

            Args.
            - x [Tensor]: Input tensor of shape [batch_size, seq_length, dim_model]

            Returns:
            - embeds [Tuple[Tensor, Tensor]]: Rotational embeddings
        '''

        (bs, seq_len, d_model), device = x.shape, x.device

        index = torch.arange(seq_len).to(device).unsqueeze(-1)
        sin = torch.sin(index * self.angle)
        cos = torch.cos(index * self.angle)

        return cos, sin

    def _rot_embed(self, x : Tensor, pos_embed : Tuple[Tensor, Tensor]) -> Tensor:
        '''
            Add rotatory positional embedding to input tensor.

            Args:
            - x [Tensor]: Input tensor of shape [batch_size, seq_length, dim_model]

            Returns:
            - x [Tensor]: Output tensor with added positional embeddings
        '''

        cos, sin = pos_embed

        rot_x = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1)
        rot_x = rearrange(rot_x, '... p d -> ... (p d)')

        return x * cos + rot_x * sin
        

        


       
