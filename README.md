# Retention Network in Easy PyTorch

A simple and concise implementation of [Retentive Networks](https://arxiv.org/abs/2307.08621) as introduced in *Retentive Network: A Successor to Transformer for Large Language Models* (2023).

# Usage

Basic usage of the RetNet model:

```python
import torch
from src.retnet import RetNet

batch_size = 2
seq_length = 1024
dim_model = 512

model = RetNet(
    num_layer = 6,
    num_heads = 8,
    dim_model = dim_model,
    dropout = 0.1,
    value_factor = 2,
    msr_gate_fn = 'gelu',
    mlp_gate_fn = 'gelu',
    mlp_mult = 4,
    mlp_bias = True,
).cuda()

x = torch.randint(0, 100, (batch_size, seq_length, dim_model), dtype=torch.float32)

# Use num_chunks parameter to switch between the parallel and recurrent forward passes.
parallel_forward  = model(x, attn_mask='causal', num_chunks = None)
recurrent_forward = model(x, attn_mask='causal', num_chunks = 8)

# The two formulations should be consistent
assert torch.allclose(parallel_forward, recurrent_forward)
```

Model now has support for both attention mask and retention matrix normalizations as described in the paper.

```python

# This is the default model behavior
no_nomalization_forward = model(x, attn_mask='causal', normalize_attn=False, normalize_retn=False, num_chunks = None)

# Normalization can be switched on independently
# ! Please NOTE that normalize_attn=True currently breaks consistency between parallel and recurrent forward
only_attn_norm_forward = model(x, attn_mask='causal', normalize_attn=True, normalize_retn=False, num_chunks = None) 
only_retn_norm_forward = model(x, attn_mask='causal', normalize_attn=False, normalize_retn=True, num_chunks = None)
```

# Known Issues

- Consistency between `parallel` and `recurrent` formulation breaks down if attention mask is normalized as suggested in the original paper (normalization #2 in the paper).
- Consistency between `parallel` and `recurrent` formulation seems unstable when a large number of `chunks` is used. This is possibly due to numerical errors that accumulate.

# Citation

This code is based on the official [authors' implementation](https://aka.ms/retnet), which is part of the larger `torchscale` codebase.

```bibtex
@article{sun2023retentive,
  title={Retentive Network: A Successor to Transformer for Large Language Models},
  author={Sun, Yutao and Dong, Li and Huang, Shaohan and Ma, Shuming and Xia, Yuqing and Xue, Jilong and Wang, Jianyong and Wei, Furu},
  journal={arXiv preprint arXiv:2307.08621},
  year={2023}
}
```