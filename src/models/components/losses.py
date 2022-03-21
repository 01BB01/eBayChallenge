import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.distributed import gather_tensor_along_batch_with_backward, get_rank


class ContrastiveLoss(nn.Module):
    """
    This is a generic contrastive loss typically used for pretraining. No modality
    assumptions are made here.
    """

    def __init__(self, temperature: float = 0.07, learnable: bool = True):
        super().__init__()
        if learnable:
            self.temperature = nn.Parameter(torch.FloatTensor((temperature,)))
        else:
            self.temperature = temperature

    def forward(self, embedding_1, embedding_2):
        per_gpu_batch_size = embedding_1.size(0)

        embedding_1 = F.normalize(embedding_1, p=2, dim=-1)
        embedding_2 = F.normalize(embedding_2, p=2, dim=-1)

        embedding_1_all_gpus = gather_tensor_along_batch_with_backward(embedding_1)
        embedding_2_all_gpus = gather_tensor_along_batch_with_backward(embedding_2)

        logits_1 = (
            torch.matmul(embedding_1, embedding_2_all_gpus.transpose(0, 1)) / self.temperature
        )
        logits_2 = (
            torch.matmul(embedding_2, embedding_1_all_gpus.transpose(0, 1)) / self.temperature
        )
        labels = per_gpu_batch_size * get_rank() + torch.arange(
            per_gpu_batch_size, device=embedding_1.device
        )

        loss_1 = F.cross_entropy(logits_1, labels)
        loss_2 = F.cross_entropy(logits_2, labels)

        return (loss_1 + loss_2) / 2


class MarginSoftmaxLoss(nn.Module):
    def __init__(self, s: float = 8, m: float = 0.2, m_type: str = "cos", **kwargs):
        super().__init__()
        self.s = s
        self.m = m
        self.m_type = m_type

    def compute_margin_logits(self, logits, labels):
        if self.m_type == "cos":
            y_onehot = torch.zeros_like(logits)
            y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
            margin_logits = self.s * (logits - y_onehot)
        else:
            y_onehot = torch.zeros_like(logits)
            y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
            arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
            logits = torch.cos(arc_logits + y_onehot)
            margin_logits = self.s * logits

        return margin_logits

    def forward(self, logits, labels):
        margin_logits = self.compute_margin_logits(logits, labels)
        loss_ce = F.cross_entropy(margin_logits, labels)

        return loss_ce

    def extra_repr(self) -> str:
        return "s={}, m={}, m_type:{}".format(self.s, self.m, self.m_type)


def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(
            -gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits))
        )

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


class ClassBalancedLoss(nn.Module):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """

    def __init__(self, samples_per_cls, no_of_classes, loss_type="focal", beta=0.9999, gamma=2.0):
        super().__init__()
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        self.weights = weights / np.sum(weights) * no_of_classes
        self.no_of_classes = no_of_classes
        self.loss_type = loss_type
        self.gamma = gamma

    def forward(self, logits, labels):

        labels_one_hot = F.one_hot(labels, self.no_of_classes).float()

        weights = torch.tensor(self.weights).float().to(logits.device)
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.no_of_classes)

        if self.loss_type == "focal":
            cb_loss = focal_loss(labels_one_hot, logits, weights, self.gamma)
        elif self.loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(
                input=logits, target=labels_one_hot, weight=weights
            )
        elif self.loss_type == "softmax":
            pred = logits.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
        return cb_loss


class SupervisedContrastiveLoss(nn.Module):
    """
    Implementation of the loss described in the paper Supervised Contrastive Learning :
    https://arxiv.org/abs/2004.11362
    """

    def __init__(self, temperature: float = 0.07, learnable: bool = True):
        super().__init__()
        if learnable:
            self.temperature = nn.Parameter(torch.FloatTensor((temperature,)))
        else:
            self.temperature = temperature

    def forward(self, projections, targets):
        device = projections.device
        per_gpu_batch_size = projections.size(0)
        projections = F.normalize(projections, p=2, dim=-1)

        all_projections = gather_tensor_along_batch_with_backward(projections)  # GN
        all_targets = gather_tensor_along_batch_with_backward(targets)

        dot_product_tempered = (
            torch.mm(projections, all_projections.T) / self.temperature
        )  # N x GN
        # Minus max for numerical stability with exponential.
        # Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = torch.exp(
            dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]
        )
        exp_dot_tempered = exp_dot_tempered + 1e-8
        # N x GN

        mask_similar_class = (
            targets.unsqueeze(-1).eq(all_targets.unsqueeze(0)).to(device).float()
        )  # N x GN
        mask_anchor_out = 1 - torch.eye(all_projections.size(0)).to(device)  # GN x GN
        rank = get_rank()
        start = rank * per_gpu_batch_size
        end = start + per_gpu_batch_size
        mask_anchor_out = mask_anchor_out[start:end]  # N x GN

        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(
            exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True))
        )
        supervised_contrastive_loss_per_sample = (
            torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        )
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss
