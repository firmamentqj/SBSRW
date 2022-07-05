# to import files from parent dir
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ops import mvctosvc
from util import batch_tensor, unbatch_tensor
import torch
import numpy as np
from torch import nn
from torch._six import inf
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torchvision.models as models


class ViewMaxAgregate(nn.Module):
    def __init__(self,  model):
        super().__init__()
        self.model = model

    def forward(self, mvimages):
        pooled_view = self.model(mvimages)
        return pooled_view


class ViewAvgAgregate(nn.Module):
    def __init__(self,  model):
        super().__init__()
        self.model = model

    def forward(self, mvimages):
        B, M, C, H, W = mvimages.shape
        pooled_view = torch.mean(unbatch_tensor(self.model(batch_tensor(
            mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True), dim=1)
        return pooled_view.squeeze()


class UnrolledDictModel(nn.Module):
    "a helper class that unroll pytorch models that return dictionaries instead of tensors"

    def __init__(self,  model, keyword="out"):
        super().__init__()
        self.model = model
        self.keyword = keyword

    def forward(self, x):
        return self.model(x)[self.keyword]


class ResNet(nn.Module):
    def __init__(self,  model, agr_type="max", feat_dim=512, num_classes=1000):
        super().__init__()
        self.agr_type = agr_type
        self.fc = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, num_classes)
        )
        if self.agr_type == "max":
            self.aggregation_model = ViewMaxAgregate(model=model)
        elif self.agr_type == "mean":
            self.aggregation_model = ViewAvgAgregate(model=model)

    def forward(self, mvimages):
        pooled_view = self.aggregation_model(mvimages)
        predictions = self.fc(pooled_view)
        return predictions, pooled_view


class AttentionModel(nn.Module):
    def __init__(self, hidden_layer=380):
        super(AttentionModel, self).__init__()

        self.attn_hidden_layer = hidden_layer
        self.net = nn.Sequential(nn.Conv2d(512, self.attn_hidden_layer, kernel_size=1),
                                 nn.Conv2d(self.attn_hidden_layer, 1, kernel_size=1))

    def forward(self, x):
        attn_mask = self.net(x)
        attn_mask = attn_mask.view(attn_mask.size(0), -1)
        attn_mask = nn.Softmax(dim=1)(attn_mask)
        attn_mask = attn_mask.view(attn_mask.size(0), 1, x.size(2), x.size(3))
        x_attn = x * attn_mask
        x = x + x_attn
        return x, attn_mask


class EncoderCNN(nn.Module):
    def __init__(self, out_size=300, pretrained=True, attention=True):
        super(EncoderCNN, self).__init__()
        # Embedd images in a 1000 space
        vgg_aux = models.vgg16_bn(pretrained)
        self.cnn_features = vgg_aux.features

        # Attention model
        self.attention = attention

        self.attn = AttentionModel()

        self.map = vgg_aux.classifier
        self.map._modules['6'] = nn.Linear(4096, out_size)

    def forward(self, im):
        x = self.cnn_features(im)  # Size (BS x 512 x 7 x 7)

        if self.attention:
            x, attn_mask = self.attn(x)  # Size (BS x 512)

        x = x.view(x.size(0), -1)

        x = self.map(x)
        if not self.attention:
            attn_mask = torch.zeros(7, 7)
        return x, attn_mask