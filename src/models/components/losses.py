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
        super(MarginSoftmaxLoss, self).__init__()
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
