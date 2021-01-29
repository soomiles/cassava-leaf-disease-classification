import torch
import torch.nn as nn
import timm


class CustomDeiT(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=False):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/deit:main', model_name, pretrained=pretrained)
        n_features = self.model.head.in_features
        self.model.head = nn.Linear(n_features, num_classes)
        if hasattr(self.model, 'head_dist'):
            n_features = self.model.head_dist.in_features
            self.model.head_dist = nn.Linear(n_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class CustomViT(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.head.in_features
        self.model.head = nn.Linear(n_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


def create_model(model_name: str,
                 pretrained: bool,
                 num_classes: int,
                 in_chans: int):
    if 'deit' in model_name:
        model = CustomDeiT(model_name, num_classes, pretrained)
    elif 'vit' in model_name:
        model = CustomViT(model_name, num_classes, pretrained)
    else:
        model = timm.create_model(model_name=model_name,
                                  pretrained=pretrained,
                                  num_classes=num_classes,
                                  in_chans=in_chans)
    return model