"""
Classifier architecture
"""
import torch
from torch import nn

INPUT_WIDTH = 28
INPUT_HEIGHT = 28
CHANNEL_AMOUNT = 1
CLASS_AMOUNT = 10


class DigitClassifier(nn.Module):
    """
    Classifier architecture implemented in PyTorch.
    """
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=5,
            stride=1,
            padding=4
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2
        )

        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.pool3 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            in_features=512,
            out_features=512
        )

        self.fc2 = nn.Linear(
            in_features=512,
            out_features=CLASS_AMOUNT
        )

    def forward(self, x):
        """
        Feed-Forward procedure.
        """
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)

        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x
