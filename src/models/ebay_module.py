import copy
import math
import os
from typing import Any, List

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics import F1Score, MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from torchvision.transforms import RandomChoice

from src.datamodules.components.transforms import RandomCutmix, RandomMixup
from src.utils.distributed import gather_tensor_along_batch
from src.utils.modelling import get_configured_parameters

from .components.cossim import CosSim
from .components.losses import (
    ClassBalancedLoss,
    ContrastiveLoss,
    MarginSoftmaxLoss,
    SupervisedContrastiveLoss,
)


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
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        loss_1: float = 1.0,
        loss_2: float = 1.0,
        loss_3: float = 1.0,
        linear_dim: int = 0,
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

        if linear_dim != 0:
            self.linear_feat = nn.Sequential(nn.Linear(output_dim, linear_dim), nn.GELU())
        else:
            self.linear_feat = None

        # loss function
        self.criterion_1 = torch.nn.CrossEntropyLoss()
        self.criterion_2 = torch.nn.CrossEntropyLoss()
        self.criterion_3 = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # mixup and cutmix
        mixup_transforms = []
        if mixup_alpha > 0.0:
            mixup_transforms.append(RandomMixup(1000, p=1.0, alpha=mixup_alpha))
        if cutmix_alpha > 0.0:
            mixup_transforms.append(RandomCutmix(1000, p=1.0, alpha=cutmix_alpha))
        if mixup_transforms:
            self.mixup_cutmix = RandomChoice(mixup_transforms)
        else:
            self.mixup_cutmix = None

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        if self.linear_feat is not None:
            return self.linear_feat(self.net(x))
        else:
            return self.net(x)

    def step(self, batch: Any):
        x = batch["image"]
        y_1 = batch["label_1"]
        y_2 = batch["label_2"]
        y_3 = batch["label_3"]
        if self.mixup_cutmix is not None:
            x, y_3_mix = self.mixup_cutmix(x, y_3)

        feats = self.forward(x)
        logits_1 = self.linear_1(feats)
        logits_2 = self.linear_2(feats)
        logits_3 = self.linear_3(feats)
        loss_1 = self.criterion_1(logits_1, y_1)
        loss_2 = self.criterion_2(logits_2, y_2)
        if self.mixup_cutmix is not None:
            loss_3 = self.criterion_3(logits_3, y_3_mix)
        else:
            loss_3 = self.criterion_3(logits_3, y_3)
        preds = torch.argmax(logits_3, dim=1)
        total_loss = (
            self.hparams.loss_1 * loss_1
            + self.hparams.loss_2 * loss_2
            + self.hparams.loss_3 * loss_3
        )
        return total_loss, preds, y_3

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
        if self.hparams.linear_dim != 0:
            linear_feat_params = get_configured_parameters(
                self.linear_feat,
                base_lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
                lr_multiplier=self.hparams.classifier_lr_multiplier,
            )
            return (
                net_params
                + linear_1_params
                + linear_2_params
                + linear_3_params
                + linear_feat_params
            )
        else:
            return net_params + linear_1_params + linear_2_params + linear_3_params

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(params=self.get_params())
        elif self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(params=self.get_params())
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
        if kwargs["output_dim"] != 768:
            self.linear_text = nn.Linear(kwargs["output_dim"], 768)
        else:
            self.linear_text = nn.Identity()
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
        contrastive_loss = self.contrastive_loss(self.linear_text(feats), text_feats)
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
        linear_text_params = get_configured_parameters(
            self.linear_text,
            base_lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            lr_multiplier=self.hparams.classifier_lr_multiplier,
        )
        optimizer = torch.optim.AdamW(
            params=text_net_params + linear_text_params + self.get_params()
        )
        if self.hparams.milestones is not None:
            lr_scheduler = MultiStepLR(optimizer, self.hparams.milestones)
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        return {"optimizer": optimizer}


class eBayPureContrastiveModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        text_net: torch.nn.Module,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        output_dim: int = 2048,
        warmup_steps: int = -1,
        milestones: List[int] = None,
        classifier_lr_multiplier: float = 1.0,
        **kwargs
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.text_net = text_net
        if output_dim != 768:
            self.linear_text = nn.Linear(output_dim, 768)
        else:
            self.linear_text = nn.Identity()
        self.contrastive_loss = ContrastiveLoss()
        self.rk_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.linear_text(self.net(x))

    def training_step(self, batch: Any, batch_idx: int):
        x = batch["image"]
        text = batch["text"]
        feats = self.net(x)
        text_feats = self.text_net(text)
        contrastive_loss = self.contrastive_loss(self.linear_text(feats), text_feats)

        self.log(
            "train/contrastive_loss", contrastive_loss, on_step=True, on_epoch=True, prog_bar=False
        )

        return {"loss": contrastive_loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        x = batch["image"]
        text = batch["text"]
        feats = self.forward(x)
        text_feats = self.text_net(text)
        return {"feats": feats, "text_feats": text_feats}

    @staticmethod
    def gather_filed(predictions):
        feats = []
        text_feats = []
        for pred in predictions:
            feats.extend(pred["feats"])
            text_feats.extend(pred["text_feats"])
        return torch.stack(feats), torch.stack(text_feats)

    @staticmethod
    def get_rk(
        q_ids: torch.Tensor,
        g_ids: torch.Tensor,
        q_embeddings: torch.Tensor,
        g_embeddings: torch.Tensor,
        k: torch.Tensor = torch.tensor([1, 5, 10], dtype=torch.long),
    ):
        # acclerate sort with topk
        q_embeddings = F.normalize(q_embeddings, p=2, dim=-1)
        g_embeddings = F.normalize(g_embeddings, p=2, dim=-1)
        _, indices = torch.topk(
            q_embeddings @ g_embeddings.t(), k=max(k), dim=1, largest=True, sorted=True
        )  # q * k
        pred_labels = g_ids[indices]  # q * k
        matches = pred_labels.eq(q_ids.view(-1, 1))  # q * k

        all_cmc = matches[:, : max(k)].cumsum(1)
        all_cmc[all_cmc > 1] = 1
        all_cmc = all_cmc.float().mean(0)
        ratk = all_cmc[k - 1]
        return ratk

    def validation_epoch_end(self, outputs: List[Any]):
        feats, text_feats = self.gather_filed(outputs)
        feats = gather_tensor_along_batch(feats)
        text_feats = gather_tensor_along_batch(text_feats)
        ids = torch.arange(len(feats), dtype=torch.long, device=feats.device)
        i2t_r1, i2t_r5, i2t_r10 = self.get_rk(ids, ids, feats, text_feats)
        t2i_r1, t2i_r5, t2i_r10 = self.get_rk(ids, ids, text_feats, feats)
        mean_rk = (i2t_r1 + i2t_r5 + i2t_r10 + t2i_r1 + t2i_r5 + t2i_r10) / 6
        self.rk_best.update(mean_rk)
        self.log("val/i2t_r1", i2t_r1, on_epoch=True, prog_bar=True)
        self.log("val/i2t_r5", i2t_r5, on_epoch=True, prog_bar=True)
        self.log("val/i2t_r10", i2t_r10, on_epoch=True, prog_bar=True)
        self.log("val/t2i_r1", t2i_r1, on_epoch=True, prog_bar=True)
        self.log("val/t2i_r5", t2i_r5, on_epoch=True, prog_bar=True)
        self.log("val/t2i_r10", t2i_r10, on_epoch=True, prog_bar=True)
        self.log("val/mean_rk", mean_rk, on_epoch=True, prog_bar=True)
        self.log("val/rk_best", self.rk_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        pass

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int):
        feats = self.forward(batch["image"])
        return {"feats": feats, "uuid": batch["uuid"]}

    def on_epoch_end(self):
        pass

    def get_params(self):
        net_params = get_configured_parameters(
            self.net,
            base_lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        text_net_params = get_configured_parameters(
            self.text_net,
            base_lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            lr_multiplier=self.hparams.text_lr_multiplier,
        )
        linear_text_params = get_configured_parameters(
            self.linear_text,
            base_lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            lr_multiplier=self.hparams.classifier_lr_multiplier,
        )
        return net_params + text_net_params + linear_text_params

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.AdamW(params=self.get_params())
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


class eBaySupConModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        output_dim: int = 2048,
        warmup_steps: int = -1,
        milestones: List[int] = None,
        **kwargs
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.criterion = SupervisedContrastiveLoss()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def training_step(self, batch: Any, batch_idx: int):
        image = batch["image"]
        pos_image = batch["pos_image"]
        ids = batch["id"]
        feats = self.forward(image)
        pos_feats = self.forward(pos_image)
        supcon_loss = self.criterion(torch.cat([feats, pos_feats], dim=0), ids.repeat(2))

        self.log("train/supcon_loss", supcon_loss, on_step=True, on_epoch=True, prog_bar=False)

        return {"loss": supcon_loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        pass

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        pass

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int):
        feats = self.forward(batch["image"])
        return {"feats": feats, "uuid": batch["uuid"]}

    def on_epoch_end(self):
        pass

    def get_params(self):
        net_params = get_configured_parameters(
            self.net,
            base_lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return net_params

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(params=self.get_params())
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


class eBaySupConConModule(eBayPureContrastiveModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supcon_loss = SupervisedContrastiveLoss()

    def training_step(self, batch: Any, batch_idx: int):
        x = batch["image"]
        x_pos = batch["pos_image"]
        text = batch["text"]
        ids = batch["id"]

        feats = self.forward(x)
        pos_feats = self.forward(x_pos)
        text_feats = self.text_net(text)

        contrastive_loss = self.contrastive_loss(feats, text_feats)
        supcon_loss = self.supcon_loss(torch.cat([feats, pos_feats], dim=0), ids.repeat(2))

        self.log(
            "train/contrastive_loss", contrastive_loss, on_step=True, on_epoch=True, prog_bar=False
        )
        self.log("train/supcon_loss", supcon_loss, on_step=True, on_epoch=True, prog_bar=False)

        return {"loss": contrastive_loss + supcon_loss}


class eBaySupConWithLinearModule(eBaySupConModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.linear = nn.Linear(kwargs["output_dim"], kwargs["linear_dim"])

    def forward(self, x: torch.Tensor):
        x = self.net(x)
        x = self.linear(x)
        return x

    def get_params(self):
        net_params = get_configured_parameters(
            self.net,
            base_lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        linear_params = get_configured_parameters(
            self.linear,
            base_lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            lr_multiplier=self.hparams.linear_lr_multiplier,
        )
        return net_params + linear_params


class eBayMultiModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        output_dim: int = 2048,
        warmup_steps: int = -1,
        milestones: List[int] = None,
        classifier_lr_multiplier: float = 1.0,
        optimizer: str = "adamw",
        multi_label_smoothing: bool = False,
        multi_scale_test: bool = False,
        num_classes: int = 22295,
        **kwargs
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.linear = nn.Linear(output_dim, num_classes, bias=False)

        # loss function
        if multi_label_smoothing:
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            cwd = hydra.utils.get_original_cwd()
            pos_weight = torch.Tensor(np.load(os.path.join(cwd, "src/models/pos_weight.npy")))
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = F1Score()
        self.val_acc = F1Score()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any, return_feats: bool = False):
        x = batch["image"]
        y = batch["multi_label"]

        feats = self.forward(x)
        logits = self.linear(feats)

        if self.hparams.multi_label_smoothing:
            smooth_y = y / (torch.sum(y, dim=1, keepdim=True) + 1e-8)
            loss = self.criterion(logits, smooth_y)
            _, indices = torch.topk(logits, k=10, dim=1, largest=True, sorted=True)
            preds = torch.zeros_like(y)
            preds[torch.arange(preds.size(0))[:, None], indices] = 1
        else:
            loss = self.criterion(logits, y)
            with torch.no_grad():
                preds = torch.sigmoid(logits.detach())

        if return_feats:
            return loss, preds, y.long(), feats
        return loss, preds, y.long()

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        pass

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int):
        if self.hparams.multi_scale_test:
            feats = self.net.model.forward_multiscale(batch["image"])
        else:
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
        linear_params = get_configured_parameters(
            self.linear,
            base_lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            lr_multiplier=self.hparams.classifier_lr_multiplier,
        )
        return net_params + linear_params

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(params=self.get_params())
        elif self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(params=self.get_params())
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


class eBayMultiContrastiveModule(eBayMultiModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text_net = kwargs["text_net"]
        if kwargs["output_dim"] != 768:
            self.linear_text = nn.Linear(kwargs["output_dim"], 768, bias=False)
            self.linear = nn.Linear(768, self.hparams.num_classes, bias=False)
        else:
            self.linear_text = nn.Identity()
        self.contrastive_loss = ContrastiveLoss()

    def forward(self, x: torch.Tensor):
        return self.linear_text(self.net(x))

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, feats = self.step(batch, return_feats=True)

        text_feats = self.text_net(batch["text"])
        contrastive_loss = self.contrastive_loss(feats, text_feats)

        # log train metrics
        acc = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log(
            "train/contrastive_loss", contrastive_loss, on_step=True, on_epoch=True, prog_bar=False
        )
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss + contrastive_loss}

    def get_params(self):
        net_params = get_configured_parameters(
            self.net,
            base_lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        text_net_params = get_configured_parameters(
            self.text_net,
            base_lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            lr_multiplier=self.hparams.text_lr_multiplier,
        )
        linear_params = get_configured_parameters(
            self.linear,
            base_lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            lr_multiplier=self.hparams.classifier_lr_multiplier,
        )
        linear_text_params = get_configured_parameters(
            self.linear_text,
            base_lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            lr_multiplier=self.hparams.classifier_lr_multiplier,
        )
        return net_params + linear_params + text_net_params + linear_text_params


class eBayMultiSelfDistillModule(eBayMultiModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha_i = kwargs.get("alpha_i", 0.0)
        self.alpha_t = kwargs.get("alpha_t", 0.8) - self.alpha_i
        self.alpha = 0.0
        self.momentum = kwargs.get("momentum", 0.999)
        self.total_steps = kwargs.get("total_steps", 5201 * 50)
        self.momentum_net = copy.deepcopy(self.net)
        for param in self.momentum_net.parameters():
            param.requires_grad = False
        self.momentum_linear = copy.deepcopy(self.linear)
        for param in self.momentum_linear.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def momentum_forward(self, x: torch.Tensor):
        x = self.momentum_net(x)
        x = self.momentum_linear(x)
        x = F.softmax(x, dim=1)
        return x

    @torch.no_grad()
    def momentum_update(self):
        for param_q, param_k in zip(self.net.parameters(), self.momentum_net.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
        for param_q, param_k in zip(self.linear.parameters(), self.momentum_linear.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    def step(self, batch: Any, return_feats: bool = False):
        x = batch["image"]
        y = batch["multi_label"]
        self.momentum_update()
        soft_target = self.momentum_forward(x)

        feats = self.forward(x)
        logits = self.linear(feats)

        if self.hparams.multi_label_smoothing:
            smooth_y = y / (torch.sum(y, dim=1, keepdim=True) + 1e-8)
            smooth_y = (1 - self.alpha) * smooth_y + self.alpha * soft_target
            fake_data = batch["fake"]
            smooth_y[fake_data] = soft_target[fake_data]
            loss = self.criterion(logits, smooth_y)
            _, indices = torch.topk(logits, k=10, dim=1, largest=True, sorted=True)
            preds = torch.zeros_like(y)
            preds[torch.arange(preds.size(0))[:, None], indices] = 1
        else:
            smooth_y = (1 - self.alpha) * y + self.alpha * soft_target
            loss = self.criterion(logits, smooth_y)
            with torch.no_grad():
                preds = torch.sigmoid(logits.detach())

        if return_feats:
            return loss, preds, y.long(), feats
        return loss, preds, y.long()

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

        self.alpha = self.alpha_i + self.alpha_t * (self.trainer.global_step / self.total_steps)


class eBayMultiSelfDistillLateModule(eBayMultiModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha_i = kwargs.get("alpha_i", 0.0)
        self.alpha_t = kwargs.get("alpha_t", 0.8) - self.alpha_i
        self.alpha = 0.0
        self.total_steps = kwargs.get("total_steps", 5201 * 50)
        self.momentum_update()

    @torch.no_grad()
    def momentum_forward(self, x: torch.Tensor):
        x = self.momentum_net(x)
        x = self.momentum_linear(x)
        x = F.softmax(x, dim=1)
        return x

    def momentum_update(self):
        self.momentum_net = copy.deepcopy(self.net)
        for param in self.momentum_net.parameters():
            param.requires_grad = False
        self.momentum_linear = copy.deepcopy(self.linear)
        for param in self.momentum_linear.parameters():
            param.requires_grad = False

    def training_epoch_end(self, outputs: List[Any]):
        self.momentum_update()

    def step(self, batch: Any, return_feats: bool = False):
        x = batch["image"]
        y = batch["multi_label"]
        soft_target = self.momentum_forward(x)

        feats = self.forward(x)
        logits = self.linear(feats)

        if self.hparams.multi_label_smoothing:
            smooth_y = y / (torch.sum(y, dim=1, keepdim=True) + 1e-8)
            smooth_y = (1 - self.alpha) * smooth_y + self.alpha * soft_target
            fake_data = batch["fake"]
            smooth_y[fake_data] = soft_target[fake_data]
            loss = self.criterion(logits, smooth_y)
            _, indices = torch.topk(logits, k=10, dim=1, largest=True, sorted=True)
            preds = torch.zeros_like(y)
            preds[torch.arange(preds.size(0))[:, None], indices] = 1
        else:
            smooth_y = (1 - self.alpha) * y + self.alpha * soft_target
            loss = self.criterion(logits, smooth_y)
            with torch.no_grad():
                preds = torch.sigmoid(logits.detach())

        if return_feats:
            return loss, preds, y.long(), feats
        return loss, preds, y.long()

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

        self.alpha = self.alpha_i + self.alpha_t * (self.trainer.global_step / self.total_steps)
