import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image

class ContrastiveImageDataset_SwAV(Dataset):
  """
  Dataset of images that serves two views of a subjects image and their label.
  Can delete first channel (segmentation channel) if specified
  """
  def __init__(self, data: str, labels: str, transform: transforms.Compose, mini_transform: transforms.Compose, delete_segmentation: bool, img_size: int, live_loading: bool) -> None:
    """
    data:                 Path to torch file containing images
    labels:               Path to torch file containing labels
    transform:            Compiled torchvision augmentations
    delete_segmentation:  If true, removes first channel from all images
    sim_matrix_path:      Path to file containing similarity matrix of subjects
    """
    self.data = torch.load(data)
    self.labels = torch.load(labels)
    self.transform = transform
    self.mini_transform = mini_transform
    self.live_loading = live_loading
    if delete_segmentation:
      for im in self.data:
        im[0,:,:] = 0

    self.default_transform = transforms.Compose([
      transforms.Resize(size=(img_size,img_size)),
      transforms.Lambda(lambda x : x.float())
    ])

  def __getitem__(self, indx: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Returns two augmented views of one image and its label
    """

    views = []
    for _ in range(2):
      views.append(self.transform(self.data[indx]))
    for _ in range(6):
      views.append(self.mini_transform(self.data[indx]))

    return views, self.labels[indx]

  def __len__(self) -> int:
    return len(self.data)