from contextlib import contextmanager
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_warn
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torchmetrics.functional.classification import binary_auroc, multiclass_auroc, binary_accuracy, multiclass_accuracy

from pl_bolts.models.self_supervised.evaluator import SSLEvaluator


class SSLOnlineEvaluator(Callback):  # pragma: no cover
    """Attaches a MLP for fine-tuning using the standard self-supervised protocol.

    Example::

        # your datamodule must have 2 attributes
        dm = DataModule()
        dm.num_classes = ... # the num of classes in the datamodule
        dm.name = ... # name of the datamodule (e.g. ImageNet, STL10, CIFAR10)

        # your model must have 1 attribute
        model = Model()
        model.z_dim = ... # the representation dim

        online_eval = SSLOnlineEvaluator(
            z_dim=model.z_dim
        )
    """

    def __init__(
        self,
        z_dim: int,
        drop_p: float = 0.2,
        hidden_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
        swav: bool = False,
        multimodal: bool = False,
    ):
        """
        Args:
            z_dim: Representation dimension
            drop_p: Dropout probability
            hidden_dim: Hidden dimension for the fine-tune MLP
        """
        super().__init__()

        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.drop_p = drop_p

        self.optimizer: Optional[Optimizer] = None
        self.online_evaluator: Optional[SSLEvaluator] = None
        self.num_classes: Optional[int] = None
        self.dataset: Optional[str] = None
        self.num_classes: Optional[int] = num_classes
        self.swav = swav
        self.multimodal = multimodal

        self._recovered_callback_state: Optional[Dict[str, Any]] = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        if self.num_classes is None:
            self.num_classes = trainer.datamodule.num_classes

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # must move to device after setup, as during setup, pl_module is still on cpu
        self.online_evaluator = SSLEvaluator(
            n_input=self.z_dim,
            n_classes=self.num_classes,
            p=self.drop_p,
            n_hidden=self.hidden_dim,
        ).to(pl_module.device)

        # switch fo PL compatibility reasons
        accel = (
            trainer.accelerator_connector
            if hasattr(trainer, "accelerator_connector")
            else trainer._accelerator_connector
        )
        if accel.is_distributed:
            if accel.use_ddp:
                from torch.nn.parallel import DistributedDataParallel as DDP

                self.online_evaluator = DDP(self.online_evaluator, device_ids=[pl_module.device])
            elif accel.use_dp:
                from torch.nn.parallel import DataParallel as DP

                self.online_evaluator = DP(self.online_evaluator, device_ids=[pl_module.device])
            else:
                rank_zero_warn(
                    "Does not support this type of distributed accelerator. The online evaluator will not sync."
                )

        self.optimizer = torch.optim.Adam(self.online_evaluator.parameters(), lr=1e-4)

        if self._recovered_callback_state is not None:
            self.online_evaluator.load_state_dict(self._recovered_callback_state["state_dict"])
            self.optimizer.load_state_dict(self._recovered_callback_state["optimizer_state"])

    def to_device(self, batch: Sequence, device: Union[str, torch.device]) -> Tuple[Tensor, Tensor]:

        if self.swav:
            x, y = batch
            x = x[0]
        elif self.multimodal:
            x_i, x_t, y, x_orig = batch
            x = x_orig
        else:
            _, x, y = batch

        # last input is for online eval
        x = x.to(device)
        y = y.to(device)

        return x, y

    def shared_step(
        self,
        pl_module: LightningModule,
        batch: Sequence,
    ):
        with torch.no_grad():
            with set_training(pl_module, False):
                x, y = self.to_device(batch, pl_module.device)
                representations = pl_module(x)

        # forward pass
        mlp_logits = self.online_evaluator(representations)  # type: ignore[operator]
        mlp_loss = F.cross_entropy(mlp_logits, y)

        mlp_logits_sm = mlp_logits.softmax(dim=1)
        if self.num_classes == 2:
          auc = binary_auroc(mlp_logits_sm[:, 1], y)
          acc = binary_accuracy(mlp_logits_sm[:, 1], y)
        else:
          auc = multiclass_auroc(mlp_logits_sm, y, self.num_classes)
          acc = multiclass_accuracy(mlp_logits_sm, y, self.num_classes)

        return acc, auc, mlp_loss

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int
    ) -> None:
        train_acc, train_auc, mlp_loss = self.shared_step(pl_module, batch)

        # update finetune weights
        mlp_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        pl_module.log("classifier.train.loss", mlp_loss, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("classifier.train.auc", train_auc, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("classifier.train.acc", train_acc, on_step=False, on_epoch=True, sync_dist=True)


    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        val_acc, val_auc, mlp_loss = self.shared_step(pl_module, batch)
        pl_module.log("classifier.val.loss", mlp_loss, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("classifier.val.auc", val_auc, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("classifier.val.acc", val_acc, on_step=False, on_epoch=True, sync_dist=True)

    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]) -> dict:
        return {"state_dict": self.online_evaluator.state_dict(), "optimizer_state": self.optimizer.state_dict()}

    def on_load_checkpoint(self, trainer: Trainer, pl_module: LightningModule, callback_state: Dict[str, Any]) -> None:
        self._recovered_callback_state = callback_state


@contextmanager
def set_training(module: nn.Module, mode: bool):
    """Context manager to set training mode.

    When exit, recover the original training mode.
    Args:
        module: module to set training mode
        mode: whether to set training mode (True) or evaluation mode (False).
    """
    original_mode = module.training

    try:
        module.train(mode)
        yield module
    finally:
        module.train(original_mode)
