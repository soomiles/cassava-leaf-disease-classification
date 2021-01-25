import torch
import torch.nn as nn
import torch.nn.functional as F

import timm


def create_model(model_name: str,
                 pretrained: bool,
                 num_classes: int,
                 in_chans: int):
    if 'deit' in model_name:
        assert timm.__version__ == "0.3.2"
        model = torch.hub.load('facebookresearch/deit:main', model_name, pretrained=pretrained)
        n_features = model.head.in_features
        model.head = nn.Linear(n_features, num_classes)
    else:
        model = timm.create_model(model_name=model_name,
                                  pretrained=pretrained,
                                  num_classes=num_classes,
                                  in_chans=in_chans)
    return model