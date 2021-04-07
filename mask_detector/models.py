import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnext101_32x8d


class BaseModel(nn.Module):
    def __init__(
            self, 
            num_classes: int, 
            backbone_name: str = "efficientnet-b7", 
            backbone_freeze=True
        ):
        super(BaseModel, self).__init__()
        # self.backbone = EfficientNet.from_pretrained(backbone_name, num_classes=4096)
        self.backbone = resnext101_32x8d(pretrained=True)
        self.backbone.requires_grad_(not backbone_freeze)
        # self.backbone._fc.requires_grad_(True)    # efficientnet
        self.backbone.fc = nn.Linear(in_features=2048, out_features=4096, bias=True)    # resnext
        self.classifier = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        output = self.backbone(x)
        return self.classifier(output)


class GenderClassifierModel(BaseModel):
    def __init__(self, num_classes: int = 2, backbone_freeze=True):
        super(GenderClassifierModel, self).__init__(
            num_classes,
            backbone_freeze=backbone_freeze
        )