from typing import List, Tuple, Dict
import torch

from utils.ntx_ent_loss_custom import NTXentLoss
from utils.supcon_loss_custom import SupConLoss
from utils.supcon_loss_clip_binary import BinarySupConCLIPLoss
from utils.remove_fn_loss import RemoveFNLoss
from utils.remove_fn_loss_binary import BinaryRemoveFNLoss
from models.pretraining import Pretraining


class SimCLR(Pretraining):
  """
  Lightning module for imaging SimCLR.

  Alternates training between contrastive model and online classifier.
  """
  def __init__(self, hparams):
    super().__init__(hparams)

    # Imaging
    self.initialize_imaging_encoder_and_projector()
    
    # Contrastive loss
    nclasses_train = hparams.batch_size
    nclasses_val = hparams.batch_size*2-1
    self.criterion_val = NTXentLoss(temperature=self.hparams.temperature)
    if self.hparams.loss.lower() == 'remove_fn':
      self.criterion_train = RemoveFNLoss(temperature=self.hparams.temperature)
    elif self.hparams.loss.lower() == 'binary_remove_fn':
      self.criterion_train = BinaryRemoveFNLoss(temperature=self.hparams.temperature, lambda_0=self.hparams.lambda_0)
    elif self.hparams.loss.lower() == 'supcon':
      self.criterion_train = SupConLoss(temperature=self.hparams.temperature, contrast_mode='all')
    elif self.hparams.loss.lower() == 'binary_supcon':
      self.criterion_train = BinarySupConCLIPLoss(temperature=self.hparams.temperature, lambda_0=self.hparams.lambda_0)
    else:  
      print("Using NTXentLoss as no other valid loss was provided.")
      self.criterion_train = NTXentLoss(self.hparams.temperature)
      nclasses_train = hparams.batch_size*2-1
    
    self.initialize_classifier_and_metrics(nclasses_train, nclasses_val)

    print(self.encoder_imaging)
    print(self.projector_imaging)


  def training_step(self, batch: Tuple[List[torch.Tensor], torch.Tensor], _) -> torch.Tensor:
    """
    Alternates calculation of loss for training between contrastive model and online classifier.
    """
    x0, x1, y = batch

    # Train contrastive model
    z0, embeddings = self.forward_imaging(x0)
    z1, _ = self.forward_imaging(x1)
    loss, logits, labels = self.criterion_train(z0, z1, y)

    self.log("imaging.train.loss", loss, on_epoch=True, on_step=False)
    if len(x0)==self.hparams.batch_size:
      self.calc_and_log_train_embedding_acc(logits=logits, labels=labels, modality='imaging')
      
    return {'loss':loss, 'embeddings': embeddings, 'labels': y}

  def validation_step(self, batch: Tuple[List[torch.Tensor], torch.Tensor], _) -> torch.Tensor:
    """
    Validate both contrastive model and classifier
    """
    x0, x1, y = batch
    
    # Validate contrastive model
    z0, embeddings = self.forward_imaging(x0)
    z1, _ = self.forward_imaging(x1)
    loss, logits, labels = self.criterion_val(z0, z1, y)

    self.log("imaging.val.loss", loss, on_epoch=True, on_step=False)
    if len(x0)==self.hparams.batch_size:
      self.calc_and_log_val_embedding_acc(logits=logits, labels=labels, modality='imaging')

    return {'sample_augmentation': x0, 'embeddings': embeddings, 'labels': y}
  
  def configure_optimizers(self) -> Tuple[Dict, Dict]:
    """
    Define and return optimizer and scheduler for contrastive model and online classifier. 
    Scheduler for online classifier often disabled
    """
    optimizer = torch.optim.Adam(
      [
        {'params': self.encoder_imaging.parameters()}, 
        {'params': self.projector_imaging.parameters()}
      ], lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
    
    scheduler = self.initialize_scheduler(optimizer)
    
    
    return (
      { # Contrastive
        "optimizer": optimizer, 
        "lr_scheduler": scheduler
      }
    )