import random
import csv
import copy
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
import pandas as pd

class ContrastiveTabularDataset(Dataset):
  """
  Dataset of tabular data that generates two views, one untouched and one corrupted.
  The corrupted view hsd a random fraction is replaced with values sampled 
  from the empirical marginal distribution of that value
  """
  def __init__(self, data_path: str, labels_path: str, corruption_rate: float=0.6, field_lengths_tabular: str=None, one_hot: bool=True):
    self.data = self.read_and_parse_csv(data_path)
    self.labels = torch.load(labels_path)
    self.c = corruption_rate
    self.generate_marginal_distributions(data_path)

    self.field_lengths = torch.load(field_lengths_tabular)

    self.one_hot = one_hot
  
  def read_and_parse_csv(self, path: str) -> List[List[float]]:
    """
    Does what it says on the box.
    """
    with open(path,'r') as f:
      reader = csv.reader(f)
      data = []
      for r in reader:
        r2 = [float(r1) for r1 in r]
        data.append(r2)
    return data

  def generate_marginal_distributions(self, data_path: str) -> None:
    """
    Generates empirical marginal distribution by transposing data
    """
    data_df = pd.read_csv(data_path)
    self.marginal_distributions = data_df.transpose().values.tolist()

  def get_input_size(self) -> int:
    """
    Returns the number of fields in the table. 
    Used to set the input number of nodes in the MLP
    """
    if self.one_hot:
      return int(sum(self.field_lengths))
    else:
      return len(self.data[0])

  def corrupt(self, subject: List[float]) -> List[float]:
    """
    Creates a copy of a subject, selects the indices 
    to be corrupted (determined by hyperparam corruption_rate)
    and replaces their values with ones sampled from marginal distribution
    """
    subject = copy.deepcopy(subject)

    indices = random.sample(list(range(len(subject))), int(len(subject)*self.c)) 
    for i in indices:
      subject[i] = random.sample(self.marginal_distributions[i],k=1)[0] 
    return subject

  def one_hot_encode(self, subject: torch.Tensor) -> torch.Tensor:
    """
    One-hot encodes a subject's features
    """
    out = []
    for i in range(len(subject)):
      if self.field_lengths[i] == 1:
        out.append(subject[i].unsqueeze(0))
      else:
        out.append(torch.nn.functional.one_hot(subject[i].long(), num_classes=int(self.field_lengths[i])))
    return torch.cat(out)

  def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Returns two views of a subjects features, the first element being the original subject features
    and the second element being the corrupted view. Also returns the label of the subject
    """
    corrupted_item = torch.tensor(self.corrupt(self.data[index]), dtype=torch.float)
    uncorrupted_item = torch.tensor(self.data[index], dtype=torch.float)
    if self.one_hot:
      corrupted_item = self.one_hot_encode(corrupted_item)
      uncorrupted_item = self.one_hot_encode(uncorrupted_item)
    item = uncorrupted_item, corrupted_item, torch.tensor(self.labels[index], dtype=torch.long)
    return item

  def __len__(self) -> int:
    return len(self.data)
