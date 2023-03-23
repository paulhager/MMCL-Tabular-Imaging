from typing import List, Tuple, Dict, Any

import torch
import pytorch_lightning as pl
import torchmetrics
import torchvision
from sklearn.linear_model import LogisticRegression
from lightly.models.modules import SimCLRProjectionHead
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pl_bolts.utils.self_supervised import torchvision_ssl_encoder

from models.TabularEncoder import TabularEncoder

class Pretraining(pl.LightningModule):
    
  def __init__(self, hparams) -> None:
    super().__init__()
    self.save_hyperparameters(hparams)

  def initialize_imaging_encoder_and_projector(self) -> None:
    """
    Selects appropriate resnet encoder
    """
    self.encoder_imaging = torchvision_ssl_encoder(self.hparams.model)
    self.pooled_dim = 2048 if self.hparams.model=='resnet50' else 512
    self.projector_imaging = SimCLRProjectionHead(self.pooled_dim, self.hparams.embedding_dim, self.hparams.projection_dim)

  def initialize_tabular_encoder_and_projector(self) -> None:
    self.encoder_tabular = TabularEncoder(self.hparams)
    self.projector_tabular = SimCLRProjectionHead(self.hparams.embedding_dim, self.hparams.embedding_dim, self.hparams.projection_dim)

  def initialize_classifier_and_metrics(self, nclasses_train, nclasses_val):
    """
    Initializes classifier and metrics. Takes care to set correct number of classes for embedding similarity metric depending on loss.
    """
    # Classifier
    self.estimator = None

    # Accuracy calculated against all others in batch of same view except for self (i.e. -1) and all of the other view
    self.top1_acc_train = torchmetrics.Accuracy(task='multiclass', top_k=1, num_classes=nclasses_train)
    self.top1_acc_val = torchmetrics.Accuracy(task='multiclass', top_k=1, num_classes=nclasses_val)

    self.top5_acc_train = torchmetrics.Accuracy(task='multiclass', top_k=5, num_classes=nclasses_train)
    self.top5_acc_val = torchmetrics.Accuracy(task='multiclass', top_k=5, num_classes=nclasses_val)

    task = 'binary' if self.hparams.num_classes == 2 else 'multiclass'

    self.classifier_acc_train = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
    self.classifier_acc_val = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)

    self.classifier_auc_train = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
    self.classifier_auc_val = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)


  def load_pretrained_imaging_weights(self) -> None:
    """
    Can load imaging encoder with pretrained weights from previous checkpoint/run
    """
    loaded_chkpt = torch.load(self.hparams.imaging_pretrain_checkpoint)
    state_dict = loaded_chkpt['state_dict']
    state_dict_encoder = {}
    for k in list(state_dict.keys()):
      if k.startswith('encoder_imaging.'):
        state_dict_encoder[k[len('encoder_imaging.'):]] = state_dict[k]
    _ = self.encoder_imaging.load_state_dict(state_dict_encoder, strict=True)
    print("Loaded imaging weights")
    if self.hparams.pretrained_imaging_strategy == 'frozen':
      for _, param in self.encoder_imaging.named_parameters():
        param.requires_grad = False
      parameters = list(filter(lambda p: p.requires_grad, self.encoder_imaging.parameters()))
      assert len(parameters)==0

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Generates encoding of imaging data.
    """
    z, y = self.forward_imaging(x)
    return y

  def forward_imaging(self, x: torch.Tensor) -> torch.Tensor:
    """
    Generates projection and encoding of imaging data.
    """
    y = self.encoder_imaging(x)[0]
    z = self.projector_imaging(y)
    return z, y

  def forward_tabular(self, x: torch.Tensor) -> torch.Tensor:
    """
    Generates projection and encoding of tabular data.
    """
    y = self.encoder_tabular(x).flatten(start_dim=1)
    z = self.projector_tabular(y)
    return z, y


  def calc_and_log_train_embedding_acc(self, logits, labels, modality: str) -> None:
    self.top1_acc_train(logits, labels)
    self.top5_acc_train(logits, labels)
    
    self.log(f"{modality}.train.top1", self.top1_acc_train, on_epoch=True, on_step=False)
    self.log(f"{modality}.train.top5", self.top5_acc_train, on_epoch=True, on_step=False)

  def calc_and_log_val_embedding_acc(self, logits, labels, modality: str) -> None:
    self.top1_acc_val(logits, labels)
    self.top5_acc_val(logits, labels)
    
    self.log(f"{modality}.val.top1", self.top1_acc_val, on_epoch=True, on_step=False)
    self.log(f"{modality}.val.top5", self.top5_acc_val, on_epoch=True, on_step=False)


  def training_epoch_end(self, train_step_outputs: List[Any]) -> None:
    """
    Train and log classifier
    """
    if self.current_epoch != 0 and self.current_epoch % self.hparams.classifier_freq == 0:
      embeddings, labels = self.stack_outputs(train_step_outputs)
    
      self.estimator = LogisticRegression(class_weight='balanced', max_iter=1000).fit(embeddings, labels)
      preds, probs = self.predict_live_estimator(embeddings)

      self.classifier_acc_train(preds, labels)
      self.classifier_auc_train(probs, labels)

      self.log('classifier.train.accuracy', self.classifier_acc_train, on_epoch=True, on_step=False)
      self.log('classifier.train.auc', self.classifier_auc_train, on_epoch=True, on_step=False)

  def validation_epoch_end(self, validation_step_outputs: List[torch.Tensor]) -> None:
    """
    Log an image from each validation step and calc validation classifier performance
    """
    if self.hparams.log_images:
      self.logger.log_image(key="Image Example", images=[validation_step_outputs[0]['sample_augmentation']])

    # Validate classifier
    if not self.estimator is None and self.current_epoch % self.hparams.classifier_freq == 0:
      embeddings, labels = self.stack_outputs(validation_step_outputs)
      
      preds, probs = self.predict_live_estimator(embeddings)
      
      self.classifier_acc_val(preds, labels)
      self.classifier_auc_val(probs, labels)

      self.log('classifier.val.accuracy', self.classifier_acc_val, on_epoch=True, on_step=False)
      self.log('classifier.val.auc', self.classifier_auc_val, on_epoch=True, on_step=False)


  def stack_outputs(self, outputs: List[torch.Tensor]) -> torch.Tensor:
    """
    Stack outputs from multiple steps
    """
    labels = outputs[0]['labels']
    embeddings = outputs[0]['embeddings']
    for i in range(1, len(outputs)):
      labels = torch.cat((labels, outputs[i]['labels']), dim=0)
      embeddings = torch.cat((embeddings, outputs[i]['embeddings']), dim=0)

    embeddings = embeddings.detach().cpu()
    labels = labels.cpu()

    return embeddings, labels

  def predict_live_estimator(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Predict using live estimator
    """
    preds = self.estimator.predict(embeddings)
    probs = self.estimator.predict_proba(embeddings)

    preds = torch.tensor(preds)
    probs = torch.tensor(probs)
    
    # Only need probs for positive class in binary case
    if self.hparams.num_classes == 2:
      probs = probs[:,1]

    return preds, probs


  def initialize_scheduler(self, optimizer: torch.optim.Optimizer):
    if self.hparams.scheduler == 'cosine':
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(self.hparams.dataset_length*self.hparams.cosine_anneal_mult), eta_min=0, last_epoch=-1)
    elif self.hparams.scheduler == 'anneal':
      scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.hparams.warmup_epochs, max_epochs = self.hparams.max_epochs)
    else:
      raise ValueError('Valid schedulers are "cosine" and "anneal"')
    
    return scheduler