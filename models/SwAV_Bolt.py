"""Adapted from official swav implementation: https://github.com/facebookresearch/swav."""
import os

import torch
from pytorch_lightning import LightningModule
from torch import nn

from utils.swav_loss import SWAVLoss
from pl_bolts.models.self_supervised.swav.swav_resnet import resnet18, resnet50
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay



class SwAV(LightningModule):
    def __init__(
        self,
        gpus: int,
        num_samples: int,
        batch_size: int,
        num_nodes: int = 1,
        model: str = "resnet50",
        hidden_mlp: int = 2048,
        feat_dim: int = 128,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        nmb_prototypes: int = 3000,
        freeze_prototypes_epochs: int = 1,
        temperature: float = 0.1,
        sinkhorn_iterations: int = 3,
        queue_length: int = 0,  # must be divisible by total batch-size
        queue_path: str = "queue",
        epoch_queue_starts: int = 15,
        crops_for_assign: tuple = (0, 1),
        nmb_crops: tuple = (2, 6),
        first_conv: bool = True,
        maxpool1: bool = True,
        optimizer: str = "adam",
        exclude_bn_bias: bool = False,
        start_lr: float = 0.0,
        lr: float = 1e-3,
        final_lr: float = 0.0,
        weight_decay: float = 1e-6,
        epsilon: float = 0.05,
        **kwargs
    ):
        """
        Args:
            gpus: number of gpus per node used in training, passed to SwAV module
                to manage the queue and select distributed sinkhorn
            num_nodes: number of nodes to train on
            num_samples: number of image samples used for training
            batch_size: batch size per GPU in ddp
            dataset: dataset being used for train/val
            model: encoder architecture used for pre-training
            hidden_mlp: hidden layer of non-linear projection head, set to 0
                to use a linear projection head
            feat_dim: output dim of the projection head
            warmup_epochs: apply linear warmup for this many epochs
            max_epochs: epoch count for pre-training
            nmb_prototypes: count of prototype vectors
            freeze_prototypes_epochs: epoch till which gradients of prototype layer
                are frozen
            temperature: loss temperature
            sinkhorn_iterations: iterations for sinkhorn normalization
            queue_length: set queue when batch size is small,
                must be divisible by total batch-size (i.e. total_gpus * batch_size),
                set to 0 to remove the queue
            queue_path: folder within the logs directory
            epoch_queue_starts: start uing the queue after this epoch
            crops_for_assign: list of crop ids for computing assignment
            nmb_crops: number of global and local crops, ex: [2, 6]
            first_conv: keep first conv same as the original resnet architecture,
                if set to false it is replace by a kernel 3, stride 1 conv (cifar-10)
            maxpool1: keep first maxpool layer same as the original resnet architecture,
                if set to false, first maxpool is turned off (cifar10, maybe stl10)
            optimizer: optimizer to use
            exclude_bn_bias: exclude batchnorm and bias layers from weight decay in optimizers
            start_lr: starting lr for linear warmup
            lr: learning rate
            final_lr: float = final learning rate for cosine weight decay
            weight_decay: weight decay for optimizer
            epsilon: epsilon val for swav assignments
        """
        super().__init__()
        self.save_hyperparameters()

        self.gpus = gpus
        self.num_nodes = num_nodes
        self.arch = model
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.hidden_mlp = hidden_mlp
        self.pooled_dim = hidden_mlp
        self.feat_dim = feat_dim
        self.nmb_prototypes = nmb_prototypes
        self.freeze_prototypes_epochs = freeze_prototypes_epochs
        self.sinkhorn_iterations = sinkhorn_iterations

        self.queue_length = queue_length
        self.queue_path = queue_path
        self.epoch_queue_starts = epoch_queue_starts
        self.crops_for_assign = crops_for_assign
        self.nmb_crops = nmb_crops

        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        self.optim = optimizer
        self.exclude_bn_bias = exclude_bn_bias
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        self.temperature = temperature

        self.start_lr = start_lr
        self.final_lr = final_lr
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.model = self.init_model()
        self.model.to('cuda')
        self.criterion = SWAVLoss(
            gpus=self.gpus,
            num_nodes=self.num_nodes,
            temperature=self.temperature,
            crops_for_assign=self.crops_for_assign,
            nmb_crops=self.nmb_crops,
            sinkhorn_iterations=self.sinkhorn_iterations,
            epsilon=self.epsilon,
        )
        self.use_the_queue = None
        # compute iters per epoch
        global_batch_size = self.num_nodes * self.gpus * self.batch_size if self.gpus > 0 else self.batch_size
        self.train_iters_per_epoch = self.num_samples // global_batch_size
        self.queue = None

    def setup(self, stage):
        if self.queue_length > 0:
            queue_folder = os.path.join(self.logger.log_dir, self.queue_path)
            if not os.path.exists(queue_folder):
                os.makedirs(queue_folder)

            self.queue_path = os.path.join(queue_folder, "queue" + str(self.trainer.global_rank) + ".pth")

            if os.path.isfile(self.queue_path):
                self.queue = torch.load(self.queue_path)["queue"]

    def init_model(self):
        if self.arch == "resnet18":
            backbone = resnet18
        elif self.arch == "resnet50":
            backbone = resnet50

        return backbone(
            normalize=True,
            hidden_mlp=self.hidden_mlp,
            output_dim=self.feat_dim,
            nmb_prototypes=self.nmb_prototypes,
            first_conv=self.first_conv,
            maxpool1=self.maxpool1,
        )

    def forward(self, x):
        # pass single batch from the resnet backbone
        return self.model.forward_backbone(x)

    def on_train_epoch_start(self):
        if self.queue_length > 0:
            if self.trainer.current_epoch >= self.epoch_queue_starts and self.queue is None:
                self.queue = torch.zeros(
                    len(self.crops_for_assign),
                    self.queue_length // self.gpus,  # change to nodes * gpus once multi-node
                    self.feat_dim,
                )

            if self.queue is not None:
                self.queue = self.queue.to(self.device)

        self.use_the_queue = False

    def on_train_epoch_end(self) -> None:
        if self.queue is not None:
            torch.save({"queue": self.queue}, self.queue_path)

    def on_after_backward(self):
        if self.current_epoch < self.freeze_prototypes_epochs:
            for name, p in self.model.named_parameters():
                if "prototypes" in name:
                    p.grad = None

    def shared_step(self, batch):
        v1, v2, y = batch
        inputs = [v1, v2]
        #inputs = inputs[:-1]  # remove online train/eval transforms at this point

        # 1. normalize the prototypes
        with torch.no_grad():
            w = self.model.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.model.prototypes.weight.copy_(w)

        # 2. multi-res forward passes
        embedding, output = self.model(inputs)
        embedding = embedding.detach()
        bs = inputs[0].size(0)

        # SWAV loss computation
        loss, queue, use_queue = self.criterion(
            output=output,
            embedding=embedding,
            prototype_weights=self.model.prototypes.weight,
            batch_size=bs,
            queue=self.queue,
            use_queue=self.use_the_queue,
        )
        self.queue = queue
        self.use_the_queue = use_queue
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=("bias", "bn")):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [{"params": params, "weight_decay": weight_decay}, {"params": excluded_params, "weight_decay": 0.0}]

    def configure_optimizers(self):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.weight_decay)
        else:
            params = self.parameters()

        if self.optim == "lars":
            optimizer = LARS(
                params,
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
                trust_coefficient=0.001,
            )
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]