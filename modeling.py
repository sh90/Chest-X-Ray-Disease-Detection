
import torch.nn as nn
from torchvision import models
def build_model(num_classes: int = 2, pretrained: bool = True):
    try:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, num_classes))
    return model
