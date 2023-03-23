from typing import Tuple

import torch
import torchmetrics
import pytorch_lightning as pl

from models.TabularModel import TabularModel
from models.ImagingModel import ImagingModel
from models.MultimodalModel import MultimodalModel


class Evaluator_Regression(pl.LightningModule):
  def __init__(self, hparams):
    super().__init__()
    self.save_hyperparameters(hparams)

    if self.hparams.datatype == 'imaging' or self.hparams.datatype == 'multimodal':
      self.model = ImagingModel(self.hparams)
    if self.hparams.datatype == 'tabular':
      self.model = TabularModel(self.hparams)
    if self.hparams.datatype == 'imaging_and_tabular':
      self.model = MultimodalModel(self.hparams)
    
    self.criterion = torch.nn.MSELoss()

    self.mae_train = torchmetrics.MeanAbsoluteError()
    self.mae_val = torchmetrics.MeanAbsoluteError()
    self.mae_test = torchmetrics.MeanAbsoluteError()

    self.pcc_train = torchmetrics.PearsonCorrCoef(num_outputs=hparams.num_classes)
    self.pcc_val = torchmetrics.PearsonCorrCoef(num_outputs=hparams.num_classes)
    self.pcc_test = torchmetrics.PearsonCorrCoef(num_outputs=hparams.num_classes)
    
    self.best_val_score = 0

    print(self.model)

  def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> None:
    """
    Runs test step
    """
    x, y = batch 
    y_hat = self.forward(x)
    y_hat = y_hat.detach()

    self.mae_test(y_hat, y)
    self.pcc_test(y_hat, y)

  def test_epoch_end(self, _) -> None:
    """
    Test epoch end
    """
    test_mae = self.mae_test.compute()
    test_pcc = self.pcc_test.compute()
    test_pcc_mean = torch.mean(test_pcc)

    self.log('test.mae', test_mae)
    self.log('test.pcc.mean', test_pcc_mean, metric_attribute=self.pcc_test)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Generates a prediction from a data point
    """
    y_hat = self.model(x)
    return y_hat

  def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
    """
    Train and log.
    """
    x, y = batch

    y_hat = self.forward(x)
    loss = self.criterion(y_hat, y)
    y_hat = y_hat.detach()

    self.mae_train(y_hat, y)
    self.pcc_train(y_hat, y)

    self.log('eval.train.loss', loss, on_epoch=True, on_step=False)
    self.log('eval.train.mae', self.mae_train, on_epoch=True, on_step=False)

    return loss
  
  def training_epoch_end(self, _) -> None:
    epoch_pcc_train = self.pcc_train.compute()
    epoch_pcc_train_mean = epoch_pcc_train.mean()
    self.log('eval.train.pcc.mean', epoch_pcc_train_mean, on_epoch=True, on_step=False, metric_attribute=self.pcc_train)
    self.pcc_train.reset()

  def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
    """
    Validate and log
    """
    x, y = batch

    y_hat = self.forward(x)
    loss = self.criterion(y_hat, y)
    y_hat = y_hat.detach()

    self.mae_val(y_hat, y)
    self.pcc_val(y_hat, y)
    
    self.log('eval.val.loss', loss, on_epoch=True, on_step=False)

    
  def validation_epoch_end(self, _) -> None:
    """
    Compute validation epoch metrics and check for new best values
    """
    if self.trainer.sanity_checking:
      return  

    epoch_mae_val = self.mae_val.compute()
    epoch_pcc_val = self.pcc_val.compute()
    epoch_pcc_val_mean = torch.mean(epoch_pcc_val)

    self.log('eval.val.mae', epoch_mae_val, on_epoch=True, on_step=False, metric_attribute=self.mae_val)
    self.log('eval.val.pcc.mean', epoch_pcc_val_mean, on_epoch=True, on_step=False, metric_attribute=self.pcc_val)
    
    self.best_val_score = max(self.best_val_score, epoch_pcc_val_mean)
    
    self.mae_val.reset()
    self.pcc_val.reset()

    
  def configure_optimizers(self):
    """
    Sets optimizer and scheduler.
    Must use strict equal to false because if check_val_n_epochs is > 1
    because val metrics not defined when scheduler is queried
    """
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr_eval, weight_decay=self.hparams.weight_decay_eval)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=int(10/self.hparams.check_val_every_n_epoch), min_lr=self.hparams.lr*0.0001)
    return optimizer
    
    return (
      {
        "optimizer": optimizer, 
        "lr_scheduler": {
          "scheduler": scheduler,
          "monitor": 'eval.val.loss',
          "strict": False
        }
      }
    )