from typing import Tuple
import random

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.io import read_image

from utils.utils import grab_hard_eval_image_augmentations, grab_soft_eval_image_augmentations, grab_image_augmentations

class ImageDataset(Dataset):
  """
  Dataset for the evaluation of images
  """
  def __init__(self, data: str, labels: str, delete_segmentation: bool, eval_train_augment_rate: float, img_size: int, target: str, train: bool, live_loading: bool, task: str) -> None:
    super(ImageDataset, self).__init__()
    self.train = train
    self.eval_train_augment_rate = eval_train_augment_rate
    self.live_loading = live_loading
    self.task = task

    self.data = torch.load(data)
    self.labels = torch.load(labels)

    if delete_segmentation:
      for im in self.data:
        im[0,:,:] = 0

    self.transform_train = grab_hard_eval_image_augmentations(img_size, target)
    self.transform_val = transforms.Compose([
      transforms.Resize(size=(img_size,img_size)),
      transforms.Lambda(lambda x : x.float())
    ])


  def __getitem__(self, indx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns an image for evaluation purposes.
    If training, has {eval_train_augment_rate} chance of being augmented.
    If val, never augmented.
    """
    im = self.data[indx]
    if self.live_loading:
      im = read_image(im)
      im = im / 255

    if self.train and (random.random() <= self.eval_train_augment_rate):
      im = self.transform_train(im)
    else:
      im = self.transform_val(im)
    
    label = self.labels[indx]
    return (im), label

  def __len__(self) -> int:
    return len(self.labels)
