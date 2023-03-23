import os, sys
dir2 = os.path.abspath('')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)

import unittest
import tempfile
import torch

from utils.remove_fn_loss import RemoveFNLoss
from utils.supcon_loss_clip import SupConLossCLIP

class LossTests(unittest.TestCase):

    def setUp(self):
        super(LossTests, self).setUp()

        self.temperature = 0.1
        self.threshold = 0.9

        self.out0 = torch.tensor([
            [0.2, 0.3, 0.6], 
            [-0.4, 0.2, 0.8], 
            [0.3, 0.5, -0.5]], device='cuda')
        self.out1 = torch.tensor([
            [0.25, 0.35, 0.55], 
            [-0.2, 0.1, 0.5], 
            [-0.2, 0.3, 0.1]], device='cuda')

        self.cosine_similarity_matrix = torch.tensor([
                    [1, 1, 0],
                    [1, 1, 0],
                    [0, 0, 1]], device='cuda')

        self.indices = [0,1,2]

    def teardown(self):
        pass

    def test_remove_fn(self):
        expected = torch.tensor(0.8850, device='cuda')
                    
        with tempfile.NamedTemporaryFile() as temp_file:
            cosine_similarity_matrix_path = temp_file.name

            torch.save(self.cosine_similarity_matrix, temp_file)
            
            criterion = RemoveFNLoss(
                temperature=self.temperature,
                cosine_similarity_matrix_path=cosine_similarity_matrix_path,
                threshold=self.threshold)

            loss, logits, labels = criterion(self.out0, self.out1, self.indices)

            torch.testing.assert_close(loss, expected, rtol=1e-4, atol=1e-4)

    def test_supcon_loss(self):
        expected = torch.tensor(1.9095, device='cuda')
                    
        with tempfile.NamedTemporaryFile() as temp_file:
            cosine_similarity_matrix_path = temp_file.name

            torch.save(self.cosine_similarity_matrix, temp_file)
            
            criterion = SupConLossCLIP(
                temperature=self.temperature,
                cosine_similarity_matrix_path=cosine_similarity_matrix_path,
                threshold=self.threshold)

            loss, logits_acc, labels_acc = criterion(self.out0, self.out1, self.indices)

            torch.testing.assert_close(loss, expected, rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
  unittest.main()