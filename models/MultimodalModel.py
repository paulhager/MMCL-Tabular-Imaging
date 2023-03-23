import torch
import torch.nn as nn
from collections import OrderedDict

from models.TabularModel import TabularModel
from models.ImagingModel import ImagingModel

class MultimodalModel(nn.Module):
  """
  Evaluation model for imaging and tabular data.
  """
  def __init__(self, args) -> None:
    super(MultimodalModel, self).__init__()

    self.imaging_model = ImagingModel(args)
    self.tabular_model = TabularModel(args)
    in_dim = 4096
    self.head = nn.Linear(in_dim, args.num_classes)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x_im = self.imaging_model.encoder(x[0]).squeeze()
    x_tab = self.tabular_model.encoder(x[1]).squeeze()
    x = torch.cat([x_im, x_tab], dim=1)
    x = self.head(x)
    return x