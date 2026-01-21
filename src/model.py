import torch.nn as nn
from torchvision import models

def get_model():
    model = models.resnet18(pretrained=True)

    # Binary classification: NORMAL vs PNEUMONIA
    model.fc = nn.Linear(model.fc.in_features, 1)

    return model
