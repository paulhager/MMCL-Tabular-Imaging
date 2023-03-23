from typing import Tuple
import csv

import torch
from torch.utils.data import Dataset

class TabularDataset(Dataset):
  """"
  Dataset for the evaluation of tabular data
  """
  def __init__(self, data_path: str, labels_path: str, eval_one_hot: bool=True, field_lengths_tabular: str=None):
    super(TabularDataset, self).__init__()
    self.data = self.read_and_parse_csv(data_path)
    self.labels = torch.load(labels_path)
    self.eval_one_hot = eval_one_hot
    self.field_lengths = torch.load(field_lengths_tabular)

    if self.eval_one_hot:
      for i in range(len(self.data)):
        self.data[i] = self.one_hot_encode(torch.tensor(self.data[i]))
    else:
      self.data = torch.tensor(self.data, dtype=torch.float)

  def get_input_size(self) -> int:
    """
    Returns the number of fields in the table. 
    Used to set the input number of nodes in the MLP
    """
    if self.eval_one_hot:
      return int(sum(self.field_lengths))
    else:
      return len(self.data[0])
  
  def read_and_parse_csv(self, path: str):
    """
    Does what it says on the box
    """
    with open(path,'r') as f:
      reader = csv.reader(f)
      data = []
      for r in reader:
        r2 = [float(r1) for r1 in r]
        data.append(r2)
    return data

  def one_hot_encode(self, subject: torch.Tensor) -> torch.Tensor:
    """
    One-hot encodes a subject's features
    """
    out = []
    for i in range(len(subject)):
      if self.field_lengths[i] == 1:
        out.append(subject[i].unsqueeze(0))
      else:
        out.append(torch.nn.functional.one_hot(torch.clamp(subject[i],min=0,max=self.field_lengths[i]-1).long(), num_classes=int(self.field_lengths[i])))
    return torch.cat(out)

  def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return self.data[index], self.labels[index]

  def __len__(self) -> int:
    return len(self.data)
