import torch
import torch.nn as nn

from .idwt_upsample import *
from .modulated_conv2d import *
from .multichannel_image import *
from .styled_conv2d import *


class MobileSynthesisBlock(nn.Module):
    def __init__(
        self,
        channels_in,
        channels_out,
        style_dim,
        kernel_size=3,
        conv_module=ModulatedConv2d,
    ):
        super().__init__()
        self.up = IDWTUpsaplme(channels_in, style_dim)
        self.conv1 = StyledConv2d(
            channels_in // 4,
            channels_out,
            style_dim,
            kernel_size,
            conv_module=conv_module,
        )
        self.conv2 = StyledConv2d(
            channels_out, channels_out, style_dim, kernel_size, conv_module=conv_module
        )
        self.to_img = MultichannelIamge(
            channels_in=channels_out,
            channels_out=12,
            style_dim=style_dim,
            kernel_size=1,
        )

    def forward(self, hidden, style, noise=[None, None]):
        hidden = self.up(hidden, style if style.ndim == 2 else style[:, 0, :])
        hidden = self.conv1(
            hidden, style if style.ndim == 2 else style[:, 0, :], noise=noise[0]
        )
        hidden = self.conv2(
            hidden, style if style.ndim == 2 else style[:, 1, :], noise=noise[1]
        )
        img = self.to_img(hidden, style if style.ndim == 2 else style[:, 2, :])
        return hidden, img

    def wsize(self):
        return 3
