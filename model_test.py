import unittest

import numpy as np
import torch

from model import RoPE, LM

class TestRoPE(unittest.TestCase):
    def setUp(self):
        self._rope = RoPE(embedding_dim=4)

    def testSingleHead(self):
        # test_input: BxTxD
        test_input = torch.tensor([[[1,2,3,4], [5,6,7,8]], [[9,10,11,12], [13,14,15,16]]], dtype=torch.float32)
        output = self._rope(test_input)
        expected_output = torch.tensor([[[ 1.0000,  2.0000,  3.0000,  4.0000],
                                         [-1.5058,  8.2906,  6.9297,  8.0796]],
                                        [[ 9.0000, 10.0000, 11.0000, 12.0000],
                                         [-3.9152, 19.3448, 14.8493, 16.1592]]], dtype=torch.float32)
        np.testing.assert_almost_equal(output.numpy(), expected_output.numpy(), decimal=4)

class TestTopKSample(unittest.TestCase):
    def setUp(self):
        self._model = LM()

    def testTopOneSampleV2(self):
        # test_input: BxTxvocab_size
        test_input = torch.tensor([[[1, 2, 3, 4], [5, 6, 8, 7]], [[9, 10, 11, 12], [16, 13, 14, 15]]],
                                  dtype=torch.float32)
        expected_output = torch.tensor([[2], [0]])
        output = self._model.top_k_sample_v2(logits=test_input, top_k=1, temperature=0.9)
        torch.testing.assert_close(output, expected_output)
