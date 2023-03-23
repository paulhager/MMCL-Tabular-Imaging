import os 
import sys

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from utils.utils import grab_image_augmentations, grab_wids, create_logdir
from utils.ssl_online_custom import SSLOnlineEvaluator

from datasets.ContrastiveImagingAndTabularDataset import ContrastiveImagingAndTabularDataset
from datasets.ContrastiveImageDataset import ContrastiveImageDataset
from datasets.ContrastiveImageDataset_ImageNet import ContrastiveImageDataset_ImageNet
from datasets.ContrastiveImageDataset_SwAV import ContrastiveImageDataset_SwAV
from datasets.ContrastiveTabularDataset import ContrastiveTabularDataset

from models.MultimodalSimCLR import MultimodalSimCLR
from models.MultimodalBYOL import MultimodalBYOL
from models.MultimodalSimCLR_MultipleLR import MultimodalSimCLR_MultipleLR
from models.MultimodalSimSiam import MultimodalSimSiam
from models.SimCLR import SimCLR
from models.SwAV_Bolt import SwAV
from models.BYOL_Bolt import BYOL
from models.SimSiam_Bolt import SimSiam
from models.BarlowTwins import BarlowTwins
from models.SCARF import SCARF



def load_datasets(hparams):
  if hparams.datatype == 'multimodal':
    transform = grab_image_augmentations(hparams.img_size, hparams.target)
    hparams.transform = transform.__repr__()
    train_dataset = ContrastiveImagingAndTabularDataset(
      hparams.data_train_imaging, hparams.delete_segmentation, transform, hparams.augmentation_rate, 
      hparams.data_train_tabular, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot,
      hparams.labels_train, hparams.img_size, hparams.live_loading)
    val_dataset = ContrastiveImagingAndTabularDataset(
      hparams.data_val_imaging, hparams.delete_segmentation, transform, hparams.augmentation_rate, 
      hparams.data_val_tabular, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot,
      hparams.labels_val, hparams.img_size, hparams.live_loading)
    hparams.input_size = train_dataset.get_input_size()
  elif hparams.datatype == 'imaging':
    transform = grab_image_augmentations(hparams.img_size, hparams.target, hparams.crop_scale_lower)
    hparams.transform = transform.__repr__()
    if hparams.target == 'ImageNet':
      wids_0 = grab_wids(hparams.majority_class)
      wids_1 = grab_wids(hparams.minority_class)
      train_dataset = ContrastiveImageDataset_ImageNet(
        base=hparams.data_base, wids_0=wids_0, wids_1=wids_1, augmentation_rate=1, split='train', live_loading=hparams.live_loading)
      val_dataset = ContrastiveImageDataset_ImageNet(
        base=hparams.data_base, wids_0=wids_0, wids_1=wids_1, augmentation_rate=0, split='val', live_loading=hparams.live_loading)
    #elif hparams.loss.lower() == 'swav':
    #  train_dataset = ContrastiveImageDataset_SwAV(
    #    data=hparams.data_train_imaging, labels=hparams.labels_train, 
    #    transform=transform, mini_transform=grab_image_augmentations(hparams.img_size//2, hparams.target), delete_segmentation=hparams.delete_segmentation, 
    #    img_size=hparams.img_size, live_loading=hparams.live_loading)
    #  val_dataset = ContrastiveImageDataset_SwAV(
    #    data=hparams.data_val_imaging, labels=hparams.labels_val, 
    #    transform=transform, mini_transform=grab_image_augmentations(hparams.img_size//2, hparams.target), delete_segmentation=hparams.delete_segmentation, 
    #    img_size=hparams.img_size, live_loading=hparams.live_loading)
    else:
      train_dataset = ContrastiveImageDataset(
        data=hparams.data_train_imaging, labels=hparams.labels_train, 
        transform=transform, delete_segmentation=hparams.delete_segmentation, 
        augmentation_rate=hparams.augmentation_rate, img_size=hparams.img_size, live_loading=hparams.live_loading)
      val_dataset = ContrastiveImageDataset(
        data=hparams.data_val_imaging, labels=hparams.labels_val, 
        transform=transform, delete_segmentation=hparams.delete_segmentation, 
        augmentation_rate=hparams.augmentation_rate, img_size=hparams.img_size, live_loading=hparams.live_loading)
  elif hparams.datatype == 'tabular':
    train_dataset = ContrastiveTabularDataset(hparams.data_train_tabular, hparams.labels_train, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot)
    val_dataset = ContrastiveTabularDataset(hparams.data_val_tabular, hparams.labels_val, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot)
    hparams.input_size = train_dataset.get_input_size()
  else:
    raise Exception(f'Unknown datatype {hparams.datatype}')
  return train_dataset, val_dataset


def select_model(hparams, train_dataset):
  if hparams.datatype == 'multimodal':
    if hparams.loss.lower() == 'byol':
      model = MultimodalBYOL(hparams)
    elif hparams.loss.lower() == 'simsiam':
      model = MultimodalSimSiam(hparams)
    elif hparams.multiple_lr:
      model = MultimodalSimCLR_MultipleLR(hparams)
    else:
      model = MultimodalSimCLR(hparams)
  elif hparams.datatype == 'imaging':
    if hparams.loss.lower() == 'byol':
      model = BYOL(**hparams)
    elif hparams.loss.lower() == 'simsiam':
      model = SimSiam(**hparams)
    elif hparams.loss.lower() == 'swav':
      if not hparams.resume_training:
        model = SwAV(gpus=1, nmb_crops=(2,0), num_samples=len(train_dataset),  **hparams)
      else:
        model = SwAV(**hparams)
    elif hparams.loss.lower() == 'barlowtwins':
      model = BarlowTwins(**hparams)
    else:
      model = SimCLR(hparams)
  elif hparams.datatype == 'tabular':
    model = SCARF(hparams)
  else:
    raise Exception(f'Unknown datatype {hparams.datatype}')
  return model


def pretrain(hparams, wandb_logger):
  """
  Train code for pretraining or supervised models. 
  
  IN
  hparams:      All hyperparameters
  wandb_logger: Instantiated weights and biases logger
  """
  pl.seed_everything(hparams.seed)

  # Load appropriate dataset
  train_dataset, val_dataset = load_datasets(hparams)
  
  train_loader = DataLoader(
    train_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=True, persistent_workers=True)

  val_loader = DataLoader(
    val_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=False, persistent_workers=True)

  # Create logdir based on WandB run name
  logdir = create_logdir(hparams.datatype, hparams.resume_training, wandb_logger)
  
  model = select_model(hparams, train_dataset)
  
  callbacks = []

  if hparams.online_mlp:
    model.hparams.classifier_freq = float('Inf')
    callbacks.append(SSLOnlineEvaluator(z_dim = model.pooled_dim, hidden_dim = hparams.embedding_dim, num_classes = hparams.num_classes, swav = False, multimodal = (hparams.datatype=='multimodal')))
  callbacks.append(ModelCheckpoint(filename='checkpoint_last_epoch_{epoch:02d}', dirpath=logdir, save_on_train_epoch_end=True, auto_insert_metric_name=False))
  callbacks.append(LearningRateMonitor(logging_interval='epoch'))

  trainer = Trainer.from_argparse_args(hparams, gpus=1, callbacks=callbacks, logger=wandb_logger, max_epochs=hparams.max_epochs, check_val_every_n_epoch=hparams.check_val_every_n_epoch, limit_train_batches=hparams.limit_train_batches, limit_val_batches=hparams.limit_val_batches, enable_progress_bar=hparams.enable_progress_bar)

  if hparams.resume_training:
    trainer.fit(model, train_loader, val_loader, ckpt_path=hparams.checkpoint)
  else:
    trainer.fit(model, train_loader, val_loader)