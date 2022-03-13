import torch
import torch.nn as nn
import torchvision


class TorchvisionImageEncoder(nn.Module):
    def __init__(
        self,
        name: str = "resnet50",
        pretrained: bool = True,
        zero_init_residual: bool = False,
        num_output_features: int = 1,
        pool_type: str = "avg",
    ):
        super().__init__()

        model = getattr(torchvision.models, name)(
            pretrained=pretrained, zero_init_residual=zero_init_residual
        )
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.pool = self._pool_func(pool_type, num_output_features)

    def _pool_func(self, pool_type, num_output_features):
        pool_func = nn.AdaptiveAvgPool2d if pool_type == "avg" else nn.AdaptiveMaxPool2d
        # -1 will keep the original feature size
        if num_output_features == -1:
            pool = nn.Identity()
        elif num_output_features in [1, 2, 3, 5, 7]:
            pool = pool_func((num_output_features, 1))
        elif num_output_features == 4:
            pool = pool_func((2, 2))
        elif num_output_features == 6:
            pool = pool_func((3, 2))
        elif num_output_features == 8:
            pool = pool_func((4, 2))
        elif num_output_features == 9:
            pool = pool_func((3, 3))

        return pool

    def forward(self, x):
        out = self.pool(self.model(x))
        out = torch.flatten(out, start_dim=1)
        return out
