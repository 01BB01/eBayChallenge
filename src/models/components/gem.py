import torch
import torch.nn as nn


class GeM(nn.Module):
    """Generalized Mean Pooling.
    Paper: https://arxiv.org/pdf/1711.02512.
    """

    def __init__(self, p: int = 3, dim=(2, 3), eps: float = 1e-6, learnable_p: bool = False):
        super().__init__()
        if learnable_p:
            self.p = nn.Parameter(torch.tensor(p))
        else:
            self.p = p
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=self.dim).pow(1.0 / self.p)
