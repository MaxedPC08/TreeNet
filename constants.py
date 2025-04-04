from torch import nn
import torch

MNIST_LAYERS = [[lambda : nn.Conv2d(1, 32, kernel_size=3, stride=1),
          lambda : nn.Conv2d(32, 64, kernel_size=3, stride=1)],
          
          [lambda : nn.Conv2d(64, 128, kernel_size=3, stride=1),
          lambda : nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
          lambda : nn.Conv2d(128, 256, kernel_size=3, stride=1),
          lambda : nn.MaxPool2d(kernel_size=2, padding=0)],

          [lambda : nn.Flatten(),
          lambda : nn.Linear(256 * 4 * 4, 128),
          lambda : nn.Linear(128, 10)]]

MNIST_END = lambda x: torch.softmax(x, dim=1)

class MNIST_GATE(nn.Module):
    def __init__(self, num_output = 2^(len(MNIST_LAYERS)-1)):
        super(MNIST_GATE, self).__init__()
        self.cnn = nn.Conv2d(1, 32, kernel_size=7, stride=1)
        self.fc = nn.Linear(32 * 22 * 22, num_output)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 32 * 22 * 22)
        x = torch.softmax(self.fc(x), dim=1)
        return x
    
CIFAR100_LAYERS = [[lambda: nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                    lambda: nn.BatchNorm2d(64, track_running_stats=True, momentum=0.1, eps=1e-5),
                    lambda: nn.ReLU(),
                    lambda: nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                    lambda: nn.BatchNorm2d(128, track_running_stats=True, momentum=0.1, eps=1e-5),
                    lambda: nn.ReLU()],

                   [lambda: nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                    lambda: nn.BatchNorm2d(256, track_running_stats=True, momentum=0.1, eps=1e-5),
                    lambda: nn.ReLU(),
                    lambda: nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    lambda: nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                    lambda: nn.BatchNorm2d(512, track_running_stats=True, momentum=0.1, eps=1e-5),
                    lambda: nn.ReLU(),
                    lambda: nn.MaxPool2d(kernel_size=2, stride=2, padding=0)],

                   [lambda: nn.Flatten(),
                    lambda: nn.Linear(512 * 8 * 8, 1024),
                    lambda: nn.GroupNorm(num_groups=32, num_channels=1024),  # Replace BatchNorm1d with GroupNorm
                    lambda: nn.ReLU(),
                    lambda: nn.Dropout(0.1),
                    lambda: nn.Linear(1024, 512),
                    lambda: nn.GroupNorm(num_groups=32, num_channels=512),  # Replace BatchNorm1d with GroupNorm
                    lambda: nn.ReLU(),
                    lambda: nn.Dropout(0.1),
                    lambda: nn.Linear(512, 100)]]

CIFAR100_END = lambda x: torch.softmax(x, dim=1)

class CIFAR100_GATE(nn.Module):
    def __init__(self, num_output=2 ** (len(CIFAR100_LAYERS) - 1)):
        super(CIFAR100_GATE, self).__init__()
        self.cnn = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
        self.fc = nn.Linear(64 * 32 * 32, num_output)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 64 * 32 * 32)
        x = torch.softmax(self.fc(x), dim=1)
        return x