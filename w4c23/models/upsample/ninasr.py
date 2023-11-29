"""
NinaSR for upsampling center region
https://github.com/Coloquinte/torchSR/blob/main/doc/NinaSR.md
https://github.com/Coloquinte/torchSR/blob/main/torchsr/models/ninasr.py
"""

import math
import torch.nn as nn
from .upsample import Upsample


class NinaSRUpsample(Upsample):
    def __init__(self, center_region, output_size, forecast_length, channels):
        super().__init__(center_region, forecast_length)
        self.upsample = NinaSR(channels, 26, 32, output_size // center_region)


class AttentionBlock(nn.Module):
    """
    A typical Squeeze-Excite attention block, with a local pooling instead of global
    """

    def __init__(self, n_feats, reduction=4, stride=16):
        super().__init__()
        self.body = nn.Sequential(
            nn.AvgPool2d(
                2 * stride - 1,
                stride=stride,
                padding=stride - 1,
                count_include_pad=False,
            ),
            nn.Conv2d(n_feats, n_feats // reduction, 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(n_feats // reduction, n_feats, 1, bias=True),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=stride, mode="nearest"),
        )

    def forward(self, x):
        res = self.body(x)
        if res.shape != x.shape:
            res = res[:, :, : x.shape[2], : x.shape[3]]
        return res * x


class ResBlock(nn.Module):
    def __init__(self, n_feats, mid_feats, in_scale, out_scale):
        super().__init__()

        self.in_scale = in_scale
        self.out_scale = out_scale

        m = []
        conv1 = nn.Conv2d(n_feats, mid_feats, 3, padding=1, bias=True)
        nn.init.kaiming_normal_(conv1.weight)
        nn.init.zeros_(conv1.bias)
        m.append(conv1)
        m.append(nn.ReLU(True))
        m.append(AttentionBlock(mid_feats))
        conv2 = nn.Conv2d(mid_feats, n_feats, 3, padding=1, bias=False)
        nn.init.kaiming_normal_(conv2.weight)
        # nn.init.zeros_(conv2.weight)
        m.append(conv2)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x * self.in_scale) * (2 * self.out_scale)
        res += x
        return res


class NinaSR(nn.Module):
    def __init__(
        self,
        n_channels,
        n_resblocks,
        n_feats,
        scale,
        expansion=2.0,
    ):
        super().__init__()
        self.scale = scale
        self.head = NinaSR.make_head(n_channels, n_feats)
        self.body = NinaSR.make_body(n_resblocks, n_feats, expansion)
        self.tail = NinaSR.make_tail(n_channels, n_feats, scale)

    @staticmethod
    def make_head(n_colors, n_feats):
        m_head = [
            nn.Conv2d(n_colors, n_feats, 3, padding=1, bias=False),
        ]
        return nn.Sequential(*m_head)

    @staticmethod
    def make_body(n_resblocks, n_feats, expansion):
        mid_feats = int(n_feats * expansion)
        out_scale = 4 / n_resblocks
        expected_variance = 1.0
        m_body = []
        for i in range(n_resblocks):
            in_scale = 1.0 / math.sqrt(expected_variance)
            m_body.append(ResBlock(n_feats, mid_feats, in_scale, out_scale))
            expected_variance += out_scale**2
        return nn.Sequential(*m_body)

    @staticmethod
    def make_tail(n_colors, n_feats, scale):
        m_tail = [
            nn.Conv2d(n_feats, n_colors * scale**2, 3, padding=1, bias=True),
            nn.PixelShuffle(scale),
        ]
        return nn.Sequential(*m_tail)

    def forward(self, x, scale=None):
        if scale is not None and scale != self.scale:
            raise ValueError(f"Network scale is {self.scale}, not {scale}")
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x
