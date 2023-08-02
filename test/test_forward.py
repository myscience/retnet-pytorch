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

        # Create a dummy input of correct shape
        self.dummy_input = torch.randint(0, 100, (1, 1024, 512), dtype = torch.float32)
        
    def test_parallel_no_attn_mask(self):
        
        # Run the model
        output : Tensor = self.retnet(self.dummy_input)

        # Check the output shape
        self.assertEqual(output.shape, (1, 1024, 512))

    def test_parallel_with_attn_mask(self):

        # Create a dummy attention mask
        attn_mask = torch.randint(0, 2, (1024, 1024), dtype = torch.bool)

        # Run the model
        output : Tensor = self.retnet(self.dummy_input, attn_mask = attn_mask)

        # Check the output shape
        self.assertEqual(output.shape, (1, 1024, 512))

    def test_parallel_causal_attn_mask(self):

        # Run the model
        output : Tensor = self.retnet(self.dummy_input, attn_mask = 'causal')

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

        # Create a dummy input of correct shape
        self.dummy_input = torch.randint(0, 100, (1, 1024, 512), dtype=torch.float32)

    def test_recurrent_forward_no_pad_needed(self):

        # Run the model
        output : Tensor = self.retnet(self.dummy_input, num_chunk=8)

        # Check the output shape
        self.assertEqual(output.shape, (1, 1024, 512))

    def test_recurrent_forward_with_pad_needed(self):
        
        # Run the model
        output_1 : Tensor = self.retnet(self.dummy_input, num_chunk=11)
        output_2 : Tensor = self.retnet(self.dummy_input, num_chunk=9)

        # Check the output shape
        self.assertEqual(output_1.shape, (1, 1024, 512))
        self.assertEqual(output_2.shape, (1, 1024, 512))

    def test_recurrent_forward_edge_cases(self):

        # Run the model
        output_1 : Tensor = self.retnet(self.dummy_input, num_chunk=0)
        output_2 : Tensor = self.retnet(self.dummy_input, num_chunk=1024)
        output_3 : Tensor = self.retnet(self.dummy_input, num_chunk=-1)
        output_4 : Tensor = self.retnet(self.dummy_input, num_chunk=5000)

        # Check the output shape
        self.assertEqual(output_1.shape, (1, 1024, 512))
        self.assertEqual(output_2.shape, (1, 1024, 512))
        self.assertEqual(output_3.shape, (1, 1024, 512))
        self.assertEqual(output_4.shape, (1, 1024, 512))

    def test_recurrent_attn_mask(self):

        # Create a dummy attention mask
        # NOTE: The attention mask should have shape [chunk_len, chunk_len]
        attn_mask = torch.randint(0, 2, (1024 // 8, 1024 // 8), dtype = torch.bool)

        # Run the model
        output : Tensor = self.retnet(self.dummy_input, num_chunk=8, attn_mask=attn_mask)

        # Check the output shape
        self.assertEqual(output.shape, (1, 1024, 512))

    def test_recurrent_causal_attn_mask(self):

        # Run the model
        output : Tensor = self.retnet(self.dummy_input, attn_mask = 'causal')

        # Check the output shape
        self.assertEqual(output.shape, (1, 1024, 512))

class ConsistencyTest(unittest.TestCase):

    def setUp(self) -> None:

        # Define the retention network
        self.retnet = RetNet(
            num_layer = 1,
            num_heads = 2,
            dim_model = 8,
            dropout = 0.0,
            value_factor = 2,
            msr_gate_fn = 'gelu',
            mlp_gate_fn = 'gelu',
            mlp_mult = 4,
            mlp_bias = True,
        )

        # Create a dummy input of correct shape
        self.dummy_input = torch.randint(0, 100, (1, 5, 8), dtype=torch.float32)

    def test_consistency_parallel_recurrent(self):
        # Compute the output of the model
        parallel_forward  = self.retnet(self.dummy_input, num_chunk=None)
        recurrent_forward = self.retnet(self.dummy_input, num_chunk=2)

        print(torch.abs(parallel_forward - recurrent_forward))

        # Check the consistency: the two output should match
        self.assertTrue(torch.allclose(parallel_forward, recurrent_forward))

if __name__ == '__main__':
    unittest.main()