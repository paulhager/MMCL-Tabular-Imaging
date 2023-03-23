import os

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from datasets.ImageDataset import ImageDataset
from datasets.TabularDataset import TabularDataset
from models.TabularEmbeddingModel import TabularEmbeddingModel
from models.ResnetEmbeddingModel import ResnetEmbeddingModel

def generate_embeddings(hparams):
  """
  Generates embeddings using trained models. 
  
  IN
  hparams:      All hyperparameters
  """
  pl.seed_everything(hparams.seed)
  if hparams.datatype == 'imaging' or hparams.datatype == 'multimodal':
    train_dataset = ImageDataset(hparams.data_train_eval_imaging, hparams.labels_train_eval_imaging, hparams.delete_segmentation, hparams.eval_train_augment_rate, hparams.img_size, train=False)
    val_dataset = ImageDataset(hparams.data_val_eval_imaging, hparams.labels_val_eval_imaging, hparams.delete_segmentation, hparams.eval_train_augment_rate, hparams.img_size, train=False)

    model = ResnetEmbeddingModel(hparams)
  elif hparams.datatype == 'tabular':
    train_dataset = TabularDataset(hparams.data_train_eval_tabular, hparams.labels_train_eval_tabular)
    val_dataset = TabularDataset(hparams.data_val_eval_tabular, hparams.labels_val_eval_tabular)
    hparams.input_size = train_dataset.get_input_size()

    model = TabularEmbeddingModel(hparams)
  else:
    raise Exception('argument dataset must be set to imaging, tabular or multimodal')
  
  train_loader = DataLoader(
    train_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=False, persistent_workers=True)

  val_loader = DataLoader(
    val_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=False, persistent_workers=True)

  model.eval()

  for (loader, split) in [(train_loader, 'train'), (val_loader, 'val')]:
    embeddings = []
    for batch in loader:
      batch_embeddings = model(batch[0]).detach()
      embeddings.extend(batch_embeddings)
    embeddings = torch.stack(embeddings)
    save_path = os.path.join(grab_rundir_from_checkpoint(hparams.checkpoint),f'{split}_embeddings.pt')
    torch.save(embeddings, save_path)

def grab_rundir_from_checkpoint(checkpoint):
  return os.path.dirname(checkpoint)