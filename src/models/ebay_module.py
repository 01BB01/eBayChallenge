import math
import os
from typing import Any, List

import hydra
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

from src.utils.modelling import get_configured_parameters

from .components.cossim import CosSim
from .components.losses import ClassBalancedLoss, ContrastiveLoss, MarginSoftmaxLoss


class eBayModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        output_dim: int = 2048,
        warmup_steps: int = -1,
        milestones: List[int] = None,
        label_smoothing: float = 0.0,
        classifier_lr_multiplier: float = 1.0,
        optimizer: str = "adam",
        **kwargs
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.linear_1 = nn.Linear(output_dim, 16, bias=False)
        self.linear_2 = nn.Linear(output_dim, 75, bias=False)
        self.linear_3 = nn.Linear(output_dim, 1000, bias=False)

        # loss function
        self.criterion_1 = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.criterion_2 = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.criterion_3 = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        x = batch["image"]
        y_1 = batch["label_1"]
        y_2 = batch["label_2"]
        y_3 = batch["label_3"]
        feats = self.forward(x)
        logits_1 = self.linear_1(feats)
        logits_2 = self.linear_2(feats)
        logits_3 = self.linear_3(feats)
        loss_1 = self.criterion_1(logits_1, y_1)
        loss_2 = self.criterion_2(logits_2, y_2)
        loss_3 = self.criterion_3(logits_3, y_3)
        preds = torch.argmax(logits_3, dim=1)
        return loss_1 + loss_2 + loss_3, preds, y_3

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        pass

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int):
        feats = self.forward(batch["image"])
        return {"feats": feats, "uuid": batch["uuid"]}

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        self.val_acc.reset()

    def get_params(self):
        net_params = get_configured_parameters(
            self.net,
            base_lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        linear_1_params = get_configured_parameters(
            self.linear_1,
            base_lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            lr_multiplier=self.hparams.classifier_lr_multiplier,
        )
        linear_2_params = get_configured_parameters(
            self.linear_2,
            base_lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            lr_multiplier=self.hparams.classifier_lr_multiplier,
        )
        linear_3_params = get_configured_parameters(
            self.linear_3,
            base_lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            lr_multiplier=self.hparams.classifier_lr_multiplier,
        )
        return net_params + linear_1_params + linear_2_params + linear_3_params

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(params=self.get_params())
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(params=self.get_params(), momentum=0.9)
        if self.hparams.milestones is not None:
            lr_scheduler = MultiStepLR(optimizer, self.hparams.milestones)
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        return {"optimizer": optimizer}

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu,
        using_native_amp,
        using_lbfgs,
    ):
        # update params
        optimizer.step(closure=optimizer_closure)

        # manually warm up lr without a scheduler
        if self.trainer.global_step < self.hparams.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.hparams.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr


class eBayCosFaceModule(eBayModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.linear_1 = CosSim(self.hparams.output_dim, 16)
        self.linear_2 = CosSim(self.hparams.output_dim, 75)
        self.linear_3 = CosSim(self.hparams.output_dim, 1000)

        # loss function
        s_1 = math.sqrt(2) * math.log(16 - 1)
        s_2 = math.sqrt(2) * math.log(75 - 1)
        s_3 = math.sqrt(2) * math.log(1000 - 1)
        self.criterion_1 = MarginSoftmaxLoss(
            s_1, self.hparams.ce_margin, self.hparams.ce_margin_type
        )
        self.criterion_2 = MarginSoftmaxLoss(
            s_2, self.hparams.ce_margin, self.hparams.ce_margin_type
        )
        self.criterion_3 = MarginSoftmaxLoss(
            s_3, self.hparams.ce_margin, self.hparams.ce_margin_type
        )


class eBayCBModule(eBayModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        cwd = hydra.utils.get_original_cwd()

        self.criterion_1 = ClassBalancedLoss(
            np.load(os.path.join(cwd, "src/models/class_1_nums.npy")), 16
        )
        self.criterion_2 = ClassBalancedLoss(
            np.load(os.path.join(cwd, "src/models/class_2_nums.npy")), 75
        )
        self.criterion_3 = ClassBalancedLoss(
            np.load(os.path.join(cwd, "src/models/class_3_nums.npy")), 1000
        )


class eBayContrastiveModule(eBayModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text_net = kwargs["text_net"]
        self.contrastive_loss = ContrastiveLoss()

    def step(self, batch: Any):
        x = batch["image"]
        y_1 = batch["label_1"]
        y_2 = batch["label_2"]
        y_3 = batch["label_3"]
        feats = self.forward(x)
        logits_1 = self.linear_1(feats)
        logits_2 = self.linear_2(feats)
        logits_3 = self.linear_3(feats)
        loss_1 = self.criterion_1(logits_1, y_1)
        loss_2 = self.criterion_2(logits_2, y_2)
        loss_3 = self.criterion_3(logits_3, y_3)
        class_loss = loss_1 + loss_2 + loss_3
        preds = torch.argmax(logits_3, dim=1)

        text_feats = self.text_net(batch["text"])
        contrastive_loss = self.contrastive_loss(feats, text_feats)
        return class_loss, contrastive_loss, preds, y_3

    def training_step(self, batch: Any, batch_idx: int):
        class_loss, contrastive_loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)
        self.log("train/loss", class_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log(
            "train/contrastive_loss", contrastive_loss, on_step=True, on_epoch=True, prog_bar=False
        )
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": class_loss + contrastive_loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, _, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def configure_optimizers(self):
        text_net_params = get_configured_parameters(
            self.text_net,
            base_lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            lr_multiplier=self.hparams.text_lr_multiplier,
        )
        optimizer = torch.optim.AdamW(params=text_net_params + self.get_params())
        if self.hparams.milestones is not None:
            lr_scheduler = MultiStepLR(optimizer, self.hparams.milestones)
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        return {"optimizer": optimizer}
