import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _resnet18_cifar(num_classes):
    # default resnet stem is 7x7 stride 2 + maxpool, too aggressive for 32x32
    # so swap in a 3x3 stride 1 conv and drop the maxpool
    model = models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def get_model(architecture, num_classes=10):
    architecture = architecture.lower()
    if architecture == "resnet18":
        return _resnet18_cifar(num_classes)
    if architecture == "simple_cnn":
        return SimpleCNN(num_classes)
    raise ValueError(f"unknown architecture: {architecture}")
