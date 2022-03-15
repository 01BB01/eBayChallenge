import timm
import torch
import torch.nn as nn
import torchvision


def get_timm(name, pretrained=True, **kwargs):
    names = [
        "vit_base_patch16_224",
        "vit_tiny_patch16_224",
        "vit_small_patch16_224",
        "vit_large_patch16_224",
        "swin_base_patch4_window12_384_in22k",
        "swin_base_patch4_window7_224_in22k",
        "swin_large_patch4_window12_384_in22k",
        "swin_large_patch4_window7_224_in22k",
    ]
    assert name in names
    model = timm.create_model(
        name,
        pretrained=pretrained,
        drop_rate=kwargs.get("drop_rate", 0.0),
        attn_drop_rate=kwargs.get("attn_drop_rate", 0.0),
        drop_path_rate=kwargs.get("drop_path_rate", 0.0),
    )
    return model


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

        self.is_timm = "vit" in name or "swin" in name
        if self.is_timm:
            model = get_timm(name, pretrained=pretrained)
            self.model = model
            # remove ImageNet classifier
            self.model.reset_classifier(-1)
        else:
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
        if self.is_timm:
            out = self.model(x)  # (bs, dim)
        else:
            out = self.pool(self.model(x))
            out = torch.flatten(out, start_dim=1)
        return out
