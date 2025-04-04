import torch
from torch import nn

class SimpleModel(nn.Module):
    def __init__(self, modules, end, device="cuda"):
        super(SimpleModel, self).__init__()
        modules = [layer for sublist in modules for layer in (sublist if isinstance(sublist, list) else [sublist])]

        modules = [module() for module in modules]
        self.model = nn.Sequential(*modules).to(device)
        self.end = end


    def forward(self, x):
        x = self.model(x)
        if self.end:
            x = self.end(x)
        return x
     
