from typing import Tuple, List

import torch
from torch import nn

class BinaryRemoveFNLoss(torch.nn.Module):
  """
  Loss function for multimodal contrastive learning based off of the CLIP paper.
  
  Embeddings are taken, L2 normalized and dot product between modalities is calculated to generate a cosine
  similarity between all combinations of subjects in a cross-modal fashion. Tempered by temperature.
  Loss is calculated attempting to match each subject's embeddings between the modalities i.e. the diagonal. 
  """
  def __init__(self, 
               temperature: float,
               lambda_0: float = 0.5) -> None:
    super(BinaryRemoveFNLoss, self).__init__()

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

    # Calc positive pull signal
    logits_mask = torch.eye(len(y), device=out0.device, dtype=torch.bool)
    pull = logits[logits_mask]

    # Calc negative push signal
    y_p = y.unsqueeze(0)
    fn_mask = y_p*y_p.T # is symmetric
    fn_mask.fill_diagonal_(0) # other view is always pushed
    push_0 = torch.log((exp_logits*~fn_mask).sum(1))
    push_1 = torch.log((exp_logits*~fn_mask).T.sum(1))

    # compute log_prob
    log_prob_0 = pull - push_0
    log_prob_1 = pull - push_1

    loss = self.lambda_0*(-log_prob_0).mean() + self.lambda_1*(-log_prob_1).mean()

    # log_prob_2 = torch.log((exp_logits.T*fn_mask).sum(1))
    # loss = self.lambda_0*(-log_prob).mean() + self.lambda_1*(-log_prob_2).mean()


    return loss, torch.matmul(out0, out1.T), torch.arange(len(out0), device=out0.device)