from typing import List, Tuple, Dict

import torch

from utils.ntx_ent_loss_custom import NTXentLoss
from utils.clip_loss import CLIPLoss
from utils.supcon_loss_clip_binary import BinarySupConCLIPLoss
from utils.supcon_loss_clip import SupConLossCLIP
from utils.kpositive_loss_clip import KPositiveLossCLIP
from utils.remove_fn_loss import RemoveFNLoss
from utils.remove_fn_loss_binary import BinaryRemoveFNLoss

from models.pretraining import Pretraining


class MultimodalSimCLR(Pretraining):
  """
  Lightning module for multimodal SimCLR.
  """
  def __init__(self, hparams):
    super().__init__(hparams)

    # Imaging
    self.initialize_imaging_encoder_and_projector()
    
    if self.hparams.imaging_pretrain_checkpoint:
      self.load_pretrained_imaging_weights()
    
    # Tabular
    self.initialize_tabular_encoder_and_projector()

    # Multimodal
    nclasses = hparams.batch_size
    self.criterion_val = CLIPLoss(temperature=self.hparams.temperature, lambda_0=self.hparams.lambda_0)
    if self.hparams.loss.lower() == 'remove_fn':
      self.criterion_train = RemoveFNLoss(temperature=self.hparams.temperature, lambda_0=self.hparams.lambda_0)
    elif self.hparams.loss.lower() == 'binary_remove_fn':
      self.criterion_train = BinaryRemoveFNLoss(temperature=self.hparams.temperature, lambda_0=self.hparams.lambda_0)
    elif self.hparams.loss.lower() == 'supcon':
      self.criterion_train = SupConLossCLIP(temperature=self.hparams.temperature, lambda_0=self.hparams.lambda_0)
    elif self.hparams.loss.lower() == 'binary_supcon':
      self.criterion_train = BinarySupConCLIPLoss(temperature=self.hparams.temperature, lambda_0=self.hparams.lambda_0)
    elif self.hparams.loss.lower() == 'kpositive':
      self.criterion_train = KPositiveLossCLIP(temperature=self.hparams.temperature, k=6, cosine_similarity_matrix_path=self.hparams.train_similarity_matrix, threshold=self.hparams.threshold)
    elif self.hparams.loss.lower() == 'clip':
      self.criterion_train = self.criterion_val
    elif self.hparams.loss.lower() == 'ntxent':
      self.criterion_train = NTXentLoss(self.hparams.temperature)
      self.criterion_val = self.criterion_train
      nclasses = hparams.batch_size*2-1
    else:
      raise ValueError('The only implemented losses currently are CLIP, NTXent, supcon, and remove_fn')

    self.initialize_classifier_and_metrics(nclasses, nclasses)

    print(f'Tabular model, multimodal: {self.encoder_tabular}\n{self.projector_tabular}')
    print(f'Imaging model, multimodal: {self.encoder_imaging}\n{self.projector_imaging}')

  def training_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], _) -> torch.Tensor:
    """
    Trains contrastive model
    """
    im_views, tab_views, y, _ = batch
    
    # Augmented views
    z0, embeddings = self.forward_imaging(im_views[1]) 
    z1, _ = self.forward_tabular(tab_views[1])
    loss, logits, labels = self.criterion_train(z0, z1, y)

    self.log(f"multimodal.train.loss", loss, on_epoch=True, on_step=False)
    if len(im_views[0])==self.hparams.batch_size:
      self.calc_and_log_train_embedding_acc(logits=logits, labels=labels, modality='multimodal')

    return {'loss':loss, 'embeddings': embeddings, 'labels': y}

  def validation_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], _) -> torch.Tensor:
    """
    Validate contrastive model
    """
    im_views, tab_views, y, original_im = batch
    
    # Unaugmented views
    z0, embeddings = self.forward_imaging(original_im)
    z1, _ = self.forward_tabular(tab_views[0])
    loss, logits, labels = self.criterion_val(z0, z1, y)

    self.log("multimodal.val.loss", loss, on_epoch=True, on_step=False)
    if len(im_views[0])==self.hparams.batch_size:
      self.calc_and_log_val_embedding_acc(logits=logits, labels=labels, modality='multimodal')

    return {'sample_augmentation': im_views[1], 'embeddings': embeddings, 'labels': y}

  def configure_optimizers(self) -> Tuple[Dict, Dict]:
    """
    Define and return optimizer and scheduler for contrastive model. 
    """
    optimizer = torch.optim.Adam(
      [
        {'params': self.encoder_imaging.parameters()}, 
        {'params': self.projector_imaging.parameters()},
        {'params': self.encoder_tabular.parameters()},
        {'params': self.projector_tabular.parameters()}
      ], lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
    
    scheduler = self.initialize_scheduler(optimizer)
    
    return (
      { # Contrastive
        "optimizer": optimizer, 
        "lr_scheduler": scheduler
      }
    )