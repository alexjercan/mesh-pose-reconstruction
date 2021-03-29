# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
# - https://arxiv.org/pdf/1512.03385.pdf
# - https://arxiv.org/pdf/2006.12250.pdf
# - https://gitlab.com/hzxie/Pix2Vox/-/blob/master/models/decoder.py

import torch
import torch.nn as nn
from models.common import TNNBlock3D


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = TNNBlock3D(in_channels=392, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.layer2 = TNNBlock3D(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.layer3 = TNNBlock3D(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.layer4 = TNNBlock3D(in_channels=32, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Sequential(
            nn.ConvTranspose3d(8, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 392, 2, 2, 2)
        return self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))


if __name__ == "__main__":
    volumes = torch.rand((2, 392, 2, 2, 2))
    decoder = Decoder()
    volumes = decoder(volumes)
    assert volumes.shape == torch.Size([2, 1, 32, 32, 32])
