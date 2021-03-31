import torch.nn as nn
from torchvision import models

class DenseBasic(nn.Module):
    def __init__(self):
        super(DenseBasic, self).__init__()
        self.dense_net = models.densenet121()
        self.dense_net.classifier = nn.Linear(1024, 18)
    
    def forward(self, source):
        output = self.dense_net(source)

        return output