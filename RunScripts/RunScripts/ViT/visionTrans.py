import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ViTModel(nn.Module):
    def __init__(self, num_classes):
        super(ViTModel, self).__init__()

        weights = ViT_B_16_Weights.DEFAULT
        self.model = vit_b_16(weights=weights)

        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)