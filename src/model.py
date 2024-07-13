import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ChannelProjector(nn.Module):
    """
    Project an arbitrary number of channels to 3 channels.
    """

    def __init__(self, input_channels):
        super(ChannelProjector, self).__init__()
        self.conv = nn.Conv2d(input_channels, 3, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ResNetFeatureExtractor(nn.Module):
    """Pretrained ResNet-18 feature extractor"""

    def __init__(self, freeze=False):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.hidden_dim = resnet.fc.in_features

        # Remove the last densee layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        if freeze:
            # Freeze all parameters
            for param in resnet.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.features(x)


class HSResNet18(nn.Module):
    def __init__(
        self,
        input_channels,
        num_outputs=6,
    ):
        super(HSResNet18, self).__init__()
        self.channel_projector = ChannelProjector(input_channels)
        self.feature_extractor = ResNetFeatureExtractor()
        self.regression_head = nn.Linear(self.feature_extractor.hidden_dim, num_outputs)

    def forward(self, x):
        x = self.channel_projector(x)
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.regression_head(x)
        return x
