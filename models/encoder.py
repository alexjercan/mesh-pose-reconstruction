# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
# - https://arxiv.org/pdf/1512.03385.pdf
# - https://arxiv.org/pdf/2006.12250.pdf
# - https://gitlab.com/hzxie/Pix2Vox/-/blob/master/models/encoder.py


import torch
import torch.nn as nn
from models.common import CNNBlock, ResBlock


class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()
        self.resnet18 = nn.Sequential(
            CNNBlock(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResBlock(in_channels=64, out_channels=64),
            ResBlock(in_channels=64, out_channels=64),
            ResBlock(in_channels=64, out_channels=64, stride=2),
            ResBlock(in_channels=64, out_channels=64),
        )
        self.layer = nn.Sequential(
            CNNBlock(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            CNNBlock(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            CNNBlock(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.resnet18(x)
        x = self.layer(x)
        return x


if __name__ == "__main__":
    layers = torch.rand((2, 9, 224, 224))
    encoder = Encoder(in_channels=9)
    layers = encoder(layers)
    assert layers.shape == torch.Size([2, 64, 7, 7])
