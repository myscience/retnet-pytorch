import torch
import unittest
from torch import Tensor

from src.retnet import RetNet

class ParallelForwardTest(unittest.TestCase):

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
    
    def test_parallel_forward(self):
        
        # Create a dummy input of correct shape
        dummy_input = torch.randint(0, 100, (1, 1024, 512))

        # Run the model
        output : Tensor = self.retnet(dummy_input)

        # Check the output shape
        self.assertEqual(output.shape, (1, 1024, 512))

class RecurrentForwardTest(unittest.TestCase):

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

    def test_recurrent_forward_no_pad_needed(self):
        # Create a dummy input of correct shape
        dummy_input = torch.randint(0, 100, (1, 1024, 512))

        # Run the model
        output : Tensor = self.retnet(dummy_input, num_chunks=8)

        # Check the output shape
        self.assertEqual(output.shape, (1, 1024, 512))

    def test_recurrent_forward_with_pad_needed(self):
        # Create a dummy input of correct shape
        dummy_input = torch.randint(0, 100, (1, 1024, 512))

        # Run the model
        output_1 : Tensor = self.retnet(dummy_input, num_chunks=11)
        output_2 : Tensor = self.retnet(dummy_input, num_chunks=9)

        # Check the output shape
        self.assertEqual(output_1.shape, (1, 1024, 512))
        self.assertEqual(output_2.shape, (1, 1024, 512))

    def test_recurrent_forward_edge_cases(self):
        # Create a dummy input of correct shape
        dummy_input = torch.randint(0, 100, (1, 1024, 512))

        # Run the model
        output_1 : Tensor = self.retnet(dummy_input, num_chunks=0)
        output_2 : Tensor = self.retnet(dummy_input, num_chunks=1024)
        output_3 : Tensor = self.retnet(dummy_input, num_chunks=-1)
        output_4 : Tensor = self.retnet(dummy_input, num_chunks=5000)

        # Check the output shape
        self.assertEqual(output_1.shape, (1, 1024, 512))
        self.assertEqual(output_2.shape, (1, 1024, 512))

if __name__ == '__main__':
    unittest.main()