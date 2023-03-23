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


class BYOL(LightningModule):
    """PyTorch Lightning implementation of Bootstrap Your Own Latent (BYOL_)_

    Paper authors: Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, \
    Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, \
    Bilal Piot, Koray Kavukcuoglu, Rémi Munos, Michal Valko.

    Args:
        lr (float, optional): optimizer learning rate. Defaults to 0.2.
        weight_decay (float, optional): optimizer weight decay. Defaults to 1.5e-6.
        warmup_epochs (int, optional): number of epochs for scheduler warmup. Defaults to 10.
        max_epochs (int, optional): maximum number of epochs for scheduler. Defaults to 1000.
        model (Union[str, torch.nn.Module], optional): base encoder architecture. Defaults to "resnet50".
        embedding_dim (int, optional): base encoder output dimension. Defaults to 2048.
        projector_hidden_dim (int, optional): projector MLP hidden dimension. Defaults to 4096.
        projector_out_dim (int, optional): projector MLP output dimension. Defaults to 256.
        initial_tau (float, optional): initial value of target decay rate used. Defaults to 0.996.

    Model implemented by:
        - `Annika Brundyn <https://github.com/annikabrundyn>`_

    Example::

        model = BYOL(num_classes=10)

        dm = CIFAR10DataModule(num_workers=0)
        dm.train_transforms = SimCLRTrainDataTransform(32)
        dm.val_transforms = SimCLREvalDataTransform(32)

        trainer = pl.Trainer()
        trainer.fit(model, datamodule=dm)

    CLI command::

        # cifar10
        python byol_module.py --gpus 1

        # imagenet
        python byol_module.py
            --gpus 8
            --dataset imagenet2012
            --data_dir /path/to/imagenet/
            --meta_dir /path/to/folder/with/meta.bin/
            --batch_size 32

    .. _BYOL: https://arxiv.org/pdf/2006.07733.pdf
    """

    def __init__(
        self,
        lr: float = 0.003,
        weight_decay: float = 1.5e-6,
        warmup_epochs: int = 10,
        max_epochs: int = 1000,
        model: Union[str, torch.nn.Module] = "resnet50",
        embedding_dim: int = 2048,
        projector_hidden_dim: int = 4096,
        projector_out_dim: int = 256,
        initial_tau: float = 0.9996,
        **kwargs: Any,
    ) -> None:

        super().__init__()
        self.save_hyperparameters()
        self.pooled_dim = 2048 if self.hparams.model=='resnet50' else 512

        self.online_network = SiameseArm(model, embedding_dim, projector_hidden_dim, projector_out_dim)
        self.target_network = deepcopy(self.online_network)

        self.weight_callback = BYOLMAWeightUpdate(initial_tau=initial_tau)

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Add callback to perform exponential moving average weight update on target network."""
        self.weight_callback.on_train_batch_end(self.trainer, self, outputs, batch, batch_idx, None)

    def forward(self, x: Tensor) -> Tensor:
        """Returns the encoded representation of a view.

        Args:
            x (Tensor): sample to be encoded
        """
        y, z, h = self.online_network(x)
        return y

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Complete training loop."""
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Complete validation loop."""
        return self._shared_step(batch, batch_idx, "val")

    def _shared_step(self, batch: Any, batch_idx: int, step: str) -> Tensor:
        """Shared evaluation step for training and validation loop."""
        img1, img2, _ = batch

        # Calculate similarity loss in each direction
        loss_12, z1 = self.calculate_loss(img1, img2)
        loss_21, z2 = self.calculate_loss(img2, img1)

        # Calculate total loss
        total_loss = loss_12 + loss_21

        # Log losses
        if step == "train":
            self.log_dict({"train_loss_12": loss_12, "train_loss_21": loss_21, "train_loss": total_loss, "variance_z1": z1.var(), "variance_z2": z2.var()})
        elif step == "val":
            self.log_dict({"val_loss_12": loss_12, "val_loss_21": loss_21, "val_loss": total_loss})
        else:
            raise ValueError(f"Step '{step}' is invalid. Must be 'train' or 'val'.")

        return total_loss

    def calculate_loss(self, v_online: Tensor, v_target: Tensor) -> Tensor:
        """Calculates similarity loss between the online network prediction of target network projection.

        Args:
            v_online (Tensor): Online network view
            v_target (Tensor): Target network view
        """
        _, z1, h1 = self.online_network(v_online)
        with torch.no_grad():
            _, z2, h2 = self.target_network(v_target)
        loss = -2 * F.cosine_similarity(h1, z2).mean()
        return loss, z1

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.hparams.warmup_epochs, max_epochs=self.hparams.max_epochs
        )
        return [optimizer], [scheduler]