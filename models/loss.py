# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import torch.nn as nn


class LossFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.entropy = nn.CrossEntropyLoss()

    def forward(self, predictions, targets):
        return self.entropy(predictions, targets)
