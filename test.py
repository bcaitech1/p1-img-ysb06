import torch
from torch import Tensor
n = 0

batch = Tensor([[ 0.9084, -0.8302]])
batch = torch.cat((batch, Tensor([[ 1.0955, -0.9059]])), dim=0)
batch = torch.cat((batch, Tensor([[ 1.5077, -1.3218]])), dim=0)
batch = torch.cat((batch, Tensor([[ 0.7238, -0.6391]])), dim=0)
batch = torch.cat((batch, Tensor([[ 1.0350, -0.9899]])), dim=0)
batch = torch.cat((batch, Tensor([[ 0.6237, -0.5912]])), dim=0)
batch = torch.cat((batch, Tensor([[ 1.4843, -1.3676]])), dim=0)
batch = torch.cat((batch, Tensor([[ 0.0470, -0.0514]])), dim=0)
batch = torch.cat((batch, Tensor([[-0.9865,  0.9710]])), dim=0)
batch = torch.cat((batch, Tensor([[-2.3072,  1.9921]])), dim=0)
batch = torch.cat((batch, Tensor([[ 1.6231, -1.3878]])), dim=0)
batch = torch.cat((batch, Tensor([[-1.8070,  1.6526]])), dim=0)
batch = torch.cat((batch, Tensor([[ 1.4390, -1.3366]])), dim=0)
batch = torch.cat((batch, Tensor([[-2.2994,  2.0113]])), dim=0)
batch = torch.cat((batch, Tensor([[ 0.0575, -0.1126]])), dim=0)
batch = torch.cat((batch, Tensor([[-0.9089,  0.8504]])), dim=0)

print(batch)
print(batch.std(dim=1))

print(batch.std(dim=1) > 1)
# print(torch.where(batch.std(dim=1) > 1, 1, 0))