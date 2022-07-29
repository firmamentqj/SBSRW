import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, Conv1d
from .blocks import *
import os


class Projection(nn.Module):
    def __init__(self,nclass):
        super().__init__()
        self.projection = Seq(*[MLP([1024,512], act='relu', norm=True, bias=True, dropout=0.5)])

        self.classifier = Seq(*[MLP([512, 256], act='relu', norm=True, bias=True, dropout=0.5),
                                MLP([256, nclass], act=None, norm=False, bias=True, dropout=0)])

    def forward(self, x):
        feat = self.projection(x)
        # print(feat.shape)
        pred = self.classifier(feat)

        return pred,feat