import torch
import torch.nn as nn
import torch.nn.functional as F


class CosSim(nn.Module):
    def __init__(self, nfeat, nclass):
        super(CosSim, self).__init__()
        self.nfeat = nfeat
        self.nclass = nclass

        codebook = torch.randn(nclass, nfeat)
        self.centroids = nn.Parameter(codebook.clone())

    def forward(self, x):
        nfeat = F.normalize(x, p=2, dim=-1)
        ncenters = F.normalize(self.centroids, p=2, dim=-1)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        return logits

    def extra_repr(self) -> str:
        return "in_features={}, n_class={}".format(self.nfeat, self.nclass)
