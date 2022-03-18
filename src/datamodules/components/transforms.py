import torch.nn as nn
import torchvision.transforms.functional as F


class DynamicPad(nn.Module):
    def __init__(self, fill=0, padding_mode="reflect"):
        super().__init__()
        if padding_mode not in ["constant", "edge", "reflect", "symmetric"]:
            raise ValueError("Padding mode should be either constant, edge, reflect or symmetric")

        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img):
        w, h = img.size
        if h == w:
            return img

        img_size = max(h, w)
        h_diff = img_size - h
        w_diff = img_size - w

        t_pad = b_pad = h_diff // 2
        if h_diff % 2 == 1:
            t_pad += 1

        l_pad = r_pad = w_diff // 2
        if w_diff % 2 == 1:
            l_pad += 1

        return F.pad(img, (l_pad, t_pad, r_pad, b_pad), self.fill, self.padding_mode)
