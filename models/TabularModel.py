from typing import Dict
from collections import OrderedDict

import torch
import torch.nn as nn

from models.TabularEncoder import TabularEncoder

class TabularModel(nn.Module):
  """
  Evaluation model for tabular trained with MLP backbone.
  """
  def __init__(self, args):
    super(TabularModel, self).__init__()

    self.encoder = TabularEncoder(args)
    self.classifier = nn.Linear(args.embedding_dim, args.num_classes)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.encoder(x)
    x = self.classifier(x)
    return x
    