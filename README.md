# Retention Network in Easy PyTorch

A simple and concise implementation of [Retentive Networks](https://arxiv.org/abs/2307.08621) as introduced in *Retentive Network: A Successor to Transformer for Large Language Models* (2023).

# Usage

Basic usage of the RetNet model:

```python
import torch
from src.retnet import RetNet

model = RetNet(
    num_layer = 6,
    num_heads = 8,
    dim_model = 512,
    dropout = 0.1,
    value_factor = 2,
    msr_gate_fn = 'gelu',
    mlp_gate_fn = 'gelu',
    mlp_mult = 4,
    mlp_bias = True,
).cuda()

x = torch.randint(0, 1000, (1, 1024)).cuda()

# Use num_chunks parameter to switch between the parallel and recurrent forward passes.
parallel_forward = model(x, num_chunks = None)
recurrent_forward = model(x, num_chunks = 8)
```

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