from typing import List, Tuple
import random
import csv
import copy

import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision.transforms import transforms
from torchvision.io import read_image

from utils.utils import grab_hard_eval_image_augmentations

class ImagingAndTabularDataset(Dataset):
  """
  Multimodal dataset that imaging and tabular data for evaluation.

  The imaging view has {eval_train_augment_rate} chance of being augmented.
  The tabular view is never augmented.
  """
  def __init__(
      self,
      data_path_imaging: str, delete_segmentation: bool, eval_train_augment_rate: float, 
      data_path_tabular: str, field_lengths_tabular: str, eval_one_hot: bool,
      labels_path: str, img_size: int, live_loading: bool, train: bool, target: str) -> None:
      
    # Imaging
    self.data_imaging = torch.load(data_path_imaging)
    self.delete_segmentation = delete_segmentation
    self.eval_train_augment_rate = eval_train_augment_rate
    self.live_loading = live_loading

    if self.delete_segmentation:
      for im in self.data_imaging:
        im[0,:,:] = 0

    self.transform_train = grab_hard_eval_image_augmentations(img_size, target)

    self.default_transform = transforms.Compose([
      transforms.Resize(size=(img_size,img_size)),
      transforms.Lambda(lambda x : x.float())
    ])

    # Tabular
    self.data_tabular = self.read_and_parse_csv(data_path_tabular)
    self.field_lengths_tabular = torch.load(field_lengths_tabular)
    self.eval_one_hot = eval_one_hot
    
    # Classifier
    self.labels = torch.load(labels_path)

    self.train = train
  
  def read_and_parse_csv(self, path_tabular: str) -> List[List[float]]:
    """
    Does what it says on the box.
    """
    with open(path_tabular,'r') as f:
      reader = csv.reader(f)
      data = []
      for r in reader:
        r2 = [float(r1) for r1 in r]
        data.append(r2)
    return data

  def get_input_size(self) -> int:
    """
    Returns the number of fields in the table. 
    Used to set the input number of nodes in the MLP
    """
    if self.eval_one_hot:
      return int(sum(self.field_lengths_tabular))
    else:
      return len(self.data[0])

  def one_hot_encode(self, subject: torch.Tensor) -> torch.Tensor:
    """
    One-hot encodes a subject's features
    """
    out = []
    for i in range(len(subject)):
      if self.field_lengths_tabular[i] == 1:
        out.append(subject[i].unsqueeze(0))
      else:
        out.append(torch.nn.functional.one_hot(torch.clamp(subject[i],min=0,max=self.field_lengths_tabular[i]-1).long(), num_classes=int(self.field_lengths_tabular[i])))
    return torch.cat(out)

  def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    im = self.data_imaging[index]
    if self.live_loading:
      im = read_image(im)
      im = im / 255

    if self.train and (random.random() <= self.eval_train_augment_rate):
      im = self.transform_train(im)
    else:
      im = self.default_transform(im)

    if self.eval_one_hot:
      tab = self.one_hot_encode(torch.tensor(self.data_tabular[index]))
    else:
      tab = torch.tensor(self.data_tabular[index], dtype=torch.float)

    label = torch.tensor(self.labels[index], dtype=torch.long)

    return (im, tab), label

  def __len__(self) -> int:
    return len(self.data_tabular)