"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
Edit: Paul Hager (paul.hager@tum.de)
Date: 19.08.2022
"""
from __future__ import print_function
from typing import Tuple, List

import torch
import torch.nn as nn


class KPositiveLossCLIP(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, cosine_similarity_matrix_path: str, temperature: float, 
                k: int=6, threshold: float=0.9):
        super(KPositiveLossCLIP, self).__init__()
        self.temperature = temperature
        self.k = k

        self.cosine_similarity_matrix = torch.load(cosine_similarity_matrix_path, map_location='cuda')
        self.cosine_similarity_matrix[self.cosine_similarity_matrix>threshold] = 1
        self.cosine_similarity_matrix = torch.threshold(self.cosine_similarity_matrix, threshold, 0)

    def forward(self, out0: torch.Tensor, out1: torch.Tensor, indices: List[int] = None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        out0 = nn.functional.normalize(out0, dim=1)
        out1 = nn.functional.normalize(out1, dim=1)
        mask = self.cosine_similarity_matrix[indices,:][:,indices]

        logits = torch.div(
            torch.cat([torch.matmul(out0, out1.T), #v1v2
                      torch.matmul(out1, out0.T)], #v2v1
                      dim=0),
            self.temperature)
            
        # for numerical stability
        # logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        # logits = logits - logits_max.detach()

        # max number of ones per row is k
        new_mask = torch.zeros_like(mask)
        for i in range(mask.shape[0]):
            indices = torch.cat([torch.tensor([i], device='cuda'), torch.randint(len(torch.flatten(mask[i].nonzero())), (self.k-1,))])
            new_mask[i, indices] = 1
        mask = new_mask

        # tile mask
        mask = mask.repeat(2,1)


        # compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (out0.shape[0]*(self.k-1))

        # loss
        loss = (-mean_log_prob_pos).mean()

        return loss, torch.matmul(out0, out1.T), torch.arange(len(out0), device=out0.device)