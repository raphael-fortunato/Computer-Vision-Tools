import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils.mutual_channel_loss import MCLoss


class BaseModel(nn.Module):
    '''
    Acts as a model wrapper for the classification models
    '''
    def __init__(self, backbone, dense):
        super().__init__()
        self.backbone = backbone
        self.maxpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = dense

    def forward(self, x):
        feat = self.backbone(x)
        out = self.maxpool(feat)
        out = out.view((out.size(0), -1))
        out = self.dense(out)
        return out, feat


def get_vgg19(path=None):
    model =models.vgg16_bn(pretrained=True)
    model.load_state_dict(torch.load('models/pretrained_vgg19_bn.pt'))
    backbone = nn.Sequential(*list(model.children())[:-2])
    dense = nn.Sequential(
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, 512),
        nn.BatchNorm1d(512),
        nn.ELU(inplace=True),
        nn.Linear(512, 16)
        )
    model = BaseModel(backbone, dense)
    if path:
        model.load_state_dict(torch.load(path))
    return model


def get_resnet50(path=None):
    model = models.resnet50(pretrained=False)
    model.load_state_dict(torch.load('models/pretrained_resnet50.pt'))
    num_ftrs = model.fc.in_features
    backbone = nn.Sequential(*list(model.children())[:-2])
    dense = nn.Linear(num_ftrs, 16)
    model = BaseModel(backbone, dense)
    if path:
        model.load_state_dict(torch.load(path))
    return model


def get_mask_rcnn(path=None):
    """
    constructs a mask r-cnn based on num_classes as output
    """
    num_classes = 2
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=False, pretrained_backbone=False)
    model.load_state_dict(torch.load('models/pretrained_maskrcnn.pt'))

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    if path:
        model.load_state_dict(torch.load(path))
    return model, in_features_mask
