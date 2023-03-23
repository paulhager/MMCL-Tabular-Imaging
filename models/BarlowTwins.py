from argparse import ArgumentParser
from copy import deepcopy
from typing import Any, Union

import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch import Tensor
from torch.nn import functional as F
from torch.optim import Adam

from pl_bolts.callbacks.byol_updates import BYOLMAWeightUpdate
from pl_bolts.models.self_supervised.byol.models import MLP, SiameseArm
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


class BarlowTwins(LightningModule):
    """
        PyTorch Lightning implementation of BarlowTwins
    """

    def __init__(
        self,
        lr: float = 0.003,
        weight_decay: float = 1.5e-6,
        warmup_epochs: int = 10,
        max_epochs: int = 1000,
        model: Union[str, torch.nn.Module] = "resnet50",
        embedding_dim: int = 2048,
        projector_hidden_dim: int = 8192,
        projector_out_dim: int = 8192,
        lambda_coeff: float = 5e-3,
        **kwargs: Any,
    ) -> None:

        super().__init__()
        self.save_hyperparameters()
        self.pooled_dim = embedding_dim

        self.network = SiameseArm(model, embedding_dim, projector_hidden_dim, projector_out_dim)
        # normalization layer for the representations z1 and z2
        self.bn = torch.nn.BatchNorm1d(projector_out_dim, affine=False)

    def forward(self, x: Tensor) -> Tensor:
        """Returns the encoded representation of a view.

        Args:
            x (Tensor): sample to be encoded
        """
        y, z, h = self.network(x)
        return y

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Complete training loop."""
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Complete validation loop."""
        return self._shared_step(batch, batch_idx, "val")
    
    def off_diagonal_ele(self, x):
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def _shared_step(self, batch: Any, batch_idx: int, step: str) -> Tensor:
        """Shared evaluation step for training and validation loop."""
        img1, img2, _ = batch

        _, z1, _ = self.network(img1)
        _, z2, _ = self.network(img2)

        z1_norm = self.bn(z1)
        z2_norm = self.bn(z2)

        cross_corr = torch.matmul(z1_norm.T, z2_norm) / self.hparams.batch_size

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        loss =  on_diag + self.hparams.lambda_coeff * off_diag

        # Log losses
        if step == "train":
            self.log_dict({"train_loss": loss})
        elif step == "val":
            self.log_dict({"val_loss": loss})
        else:
            raise ValueError(f"Step '{step}' is invalid. Must be 'train' or 'val'.")

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.hparams.warmup_epochs, max_epochs=self.hparams.max_epochs
        )
        return [optimizer], [scheduler]