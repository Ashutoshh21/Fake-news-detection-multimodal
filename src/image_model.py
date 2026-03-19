import torch
import torch.nn as nn
import torchvision.models as models


class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()

        resnet = models.resnet50(weights = None)

        # remove last classification layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        return features
