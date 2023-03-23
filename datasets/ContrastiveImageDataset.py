import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image

class ContrastiveImageDataset(Dataset):
  """
  Dataset of images that serves two views of a subjects image and their label.
  Can delete first channel (segmentation channel) if specified
  """
  def __init__(self, data: str, labels: str, transform: transforms.Compose, delete_segmentation: bool, augmentation_rate: float, img_size: int, live_loading: bool) -> None:
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
    self.augmentation_rate = augmentation_rate
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
    view_1, view_2 = self.generate_imaging_views(indx)

    return view_1, view_2, self.labels[indx]

  def __len__(self) -> int:
    return len(self.data)

  def generate_imaging_views(self, index: int) -> List[torch.Tensor]:
    """
    Generates two views of a subjects image. 
    The first is always augmented. The second has {augmentation_rate} chance to be augmented.
    """
    im = self.data[index]
    if self.live_loading:
      im = read_image(im)
      im = im / 255
    view_1 = self.transform(im)
    if random.random() < self.augmentation_rate:
      view_2 = self.transform(im)
    else:
      view_2 = self.default_transform(im)
    
    return view_1, view_2