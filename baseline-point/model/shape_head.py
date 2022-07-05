import torch
from torch.autograd import Variable
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .pointnet import *
from torch import nn
from torch.nn import Sequential as Seq


class FeatureExtractor(nn.Module):
    def __init__(self,  nclass,shape_extractor, screatch_feature_extractor=False):
        super().__init__()
        # self.features_type = features_type

        if shape_extractor == "PointNet":
            self.fe_model = PointNet(nclass, alignment=True)
        elif shape_extractor == "DGCNN":
            self.fe_model = SimpleDGCNN(nclass)
        if not screatch_feature_extractor:
            print(shape_extractor)
            load_point_ckpt(self.fe_model, shape_extractor,
                            ckpt_dir='./checkpoint')
        # self.features_order = {"logits": 0,
        #                        "post_max": 1, "transform_matrix": 2}

    def forward(self, extra_info=None):

        extra_info = extra_info.transpose(1, 2).to(
            next(self.fe_model.parameters()).device)
        features = self.fe_model(extra_info)
        # if self.features_type == "logits_trans":
        #     return torch.cat((features[0].view(c_batch_size, -1), features[2].view(c_batch_size, -1)), 1)
        # elif self.features_type == "post_max_trans":
        #     return torch.cat((features[1].view(c_batch_size, -1), features[2].view(c_batch_size, -1)), 1)
        # else:
        #     return features[self.features_order[self.features_type]].view(c_batch_size, -1)
        return features[0],features[1]
