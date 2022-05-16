import torch
import torch.nn as nn
import torch.nn.functional as F


class CosSim(nn.Module):
    def __init__(self, nfeat, nclass, temperature=1.0, learnable=False):
        super(CosSim, self).__init__()
        self.nfeat = nfeat
        self.nclass = nclass

        codebook = torch.randn(nclass, nfeat)
        self.centroids = nn.Parameter(codebook.clone())

        if learnable:
            self.temperature = nn.Parameter(torch.FloatTensor((temperature,)))
        else:
            self.temperature = temperature

    def forward(self, x):
        nfeat = F.normalize(x, p=2, dim=-1)
        ncenters = F.normalize(self.centroids, p=2, dim=-1)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        return logits / self.temperature

    def extra_repr(self) -> str:
        return "in_features={}, n_class={}".format(self.nfeat, self.nclass)
