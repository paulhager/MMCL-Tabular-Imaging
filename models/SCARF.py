from typing import List, Tuple, Dict, Any

import torch

from utils.ntx_ent_loss_custom import NTXentLoss
from models.pretraining import Pretraining


class SCARF(Pretraining):
  """
  Lightning module for SCARF pretraining. 
  """
  def __init__(self, hparams) -> None:
    super().__init__(hparams)

    self.initialize_tabular_encoder_and_projector()

    self.criterion = NTXentLoss(self.hparams.temperature)
    nclasses = hparams.batch_size*2-1

    self.initialize_classifier_and_metrics(nclasses, nclasses)
    self.pooled_dim = self.hparams.embedding_dim

    print(self.encoder_tabular)
    print(self.projector_tabular)

  def forward(self, x: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass for tabular
    """
    embeddings = self.encoder_tabular(x)
    return embeddings

  def training_step(self, batch: Tuple[List[torch.Tensor], torch.Tensor], _) -> torch.Tensor:
    """
    Trains contrastive model
    """
    x0, x1, y = batch

    # Train contrastive model
    z0, embeddings = self.forward_tabular(x0)
    z1, _ = self.forward_tabular(x1)
    loss, logits, labels = self.criterion(z0, z1)
    self.log("tabular.train.loss", loss, on_epoch=True, on_step=False)

    if len(x0)==self.hparams.batch_size:
      self.calc_and_log_train_embedding_acc(logits=logits, labels=labels, modality='tabular')

    return {'loss':loss, 'embeddings': embeddings, 'labels': y}

  def validation_step(self, batch: Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], _) -> None:
    """
    Validate both contrastive model and classifier
    """
    x0, x1, y = batch
    
    # Validate contrastive model
    z0, embeddings = self.forward_tabular(x0)
    z1, _ = self.forward_tabular(x1)
    loss, logits, labels = self.criterion(z0, z1, y)

    self.log("tabular.val.loss", loss, on_epoch=True, on_step=False)
    if len(x0)==self.hparams.batch_size:
      self.calc_and_log_val_embedding_acc(logits=logits, labels=labels, modality='tabular')

    return {'embeddings': embeddings, 'labels': y}
  
  def configure_optimizers(self) -> Tuple[Dict, Dict]:
    """
    Define and return optimizer and scheduler for contrastive model and online classifier. 
    Scheduler for online classifier often disabled
    """
    optimizer = torch.optim.Adam(
      [
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