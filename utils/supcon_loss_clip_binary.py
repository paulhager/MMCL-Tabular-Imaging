from typing import Tuple, List

import torch
from torch import nn

class BinarySupConCLIPLoss(torch.nn.Module):
  """
  Loss function for multimodal contrastive learning based off of the CLIP paper.
  
  Embeddings are taken, L2 normalized and dot product between modalities is calculated to generate a cosine
  similarity between all combinations of subjects in a cross-modal fashion. Tempered by temperature.
  Loss is calculated attempting to match each subject's embeddings between the modalities i.e. the diagonal. 
  """
  def __init__(self, 
               temperature: float,
               lambda_0: float = 0.5) -> None:
    super(BinarySupConCLIPLoss, self).__init__()

    self.temperature = temperature

    if lambda_0 > 1 or lambda_0 < 0:
      raise ValueError('lambda_0 must be a float between 0 and 1.')
    self.lambda_0 = lambda_0
    self.lambda_1 = 1-lambda_0

  def forward(self, out0: torch.Tensor, out1: torch.Tensor, y: torch.Tensor) -> Tuple:
    # normalize the embedding onto the unit hypersphere
    out0 = nn.functional.normalize(out0, dim=1)
    out1 = nn.functional.normalize(out1, dim=1)

    # Calc logits
    logits = torch.matmul(out0, out1.T) / self.temperature
    exp_logits = torch.exp(logits)

    y_p = y.unsqueeze(0)
    tp_mask = y_p*y_p.T # is symmetric
    tp_mask.fill_diagonal_(1) # other view is always pulled

    # Calc positive pull signal
    pull_0 = logits
    pull_1 = pull_0.T

    # Calc negative push signal
    push_0 = torch.log((exp_logits).sum(1, keepdim=True)+1e-6)
    push_1 = push_0.T
    
    log_prob_0 = (tp_mask*(pull_0-push_0)).sum(1) / tp_mask.sum(1)
    log_prob_1 = (tp_mask*(pull_1-push_1)).sum(1) / tp_mask.sum(1)
    loss = self.lambda_0*(-log_prob_0).mean() + self.lambda_1*(-log_prob_1).mean()
  
    return loss, logits, torch.arange(len(out0), device=out0.device)