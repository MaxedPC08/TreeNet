import torch
import torchvision

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import TreeModel
import ExpertsModel
import SimpleModel
from tests import full_mnist, full_cifar100

criterion = nn.CrossEntropyLoss()
# Print the number of parameters in the network
# total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
# print(f'Total number of trainable parameters: {total_params}')
optimizer = lambda params: optim.Adam(params, lr=0.00001)

full_cifar100(optimizer, criterion, epochs=10)
full_mnist(optimizer, criterion, epochs=1)

