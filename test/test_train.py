import torch
import unittest
from torch import Tensor
from torch.optim import AdamW

from src.retnet import RetNet

class RetNetTrainTest(unittest.TestCase):
    '''
        Test the RetNet training.
    '''
    def setUp(self) -> None:
        # Define the retention network
        self.retnet = RetNet(
            num_layer = 6,
            num_heads = 8,
            dim_model = 512,
            dropout = 0.1,
            value_factor = 2,
            msr_gate_fn = 'gelu',
            mlp_gate_fn = 'gelu',
            mlp_mult = 4,
            mlp_bias = True,
        )

        # Create a dummy input of correct shape
        self.dummy_input = torch.randint(0, 100, (1, 1024, 512), dtype=torch.float32)

        # Create a dummy target of correct shape
        self.dummy_target = torch.randint(0, 100, (1, 1024, 512), dtype=torch.float32)

        # Create a dummy optimizer
        self.optimizer = AdamW(self.retnet.parameters(), lr=1e-4)

    def test_parallel_train_loop(self) -> None:

        self.retnet.train()
        self.optimizer.zero_grad()

        # Compute the model output (no chunking triggers parallel forward)
        output = self.retnet(self.dummy_input)

        # Compute the loss
        loss = torch.nn.functional.mse_loss(output, self.dummy_target)

        # Perform a backward pass
        loss.backward()

        # Perform a single optimization step
        self.optimizer.step()

        # If you get here you win
        self.assertTrue(True)

    def test_chunked_train_loop(self) -> None:

        self.retnet.train()
        self.optimizer.zero_grad()

        # Compute the model output (use chunking)
        output = self.retnet(self.dummy_input, num_chunk=8)

        # Compute the loss
        loss = torch.nn.functional.mse_loss(output, self.dummy_target)

        # Perform a backward pass
        loss.backward()

        # Perform a single optimization step
        self.optimizer.step()

        # If you get here you win
        self.assertTrue(True)
