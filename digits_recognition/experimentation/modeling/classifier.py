"""
Classifier architecture
"""
import torch
from torch import nn


class DigitClassifier(nn.Module):
    """
    Classifier architecture implemented in PyTorch.
    """
    def __init__(self, input_height, input_width, input_channels, class_count):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
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

        flatten_size = self._get_flattened_size(input_height, input_width, input_channels)

        self.fc1 = nn.Linear(
            in_features=flatten_size,
            out_features=flatten_size
        )

        self.fc2 = nn.Linear(
            in_features=flatten_size,
            out_features=class_count
        )

    def _get_flattened_size(self, input_height, input_width, input_channels):
        dummy_input = torch.zeros(1, input_channels, input_height, input_width)
        dummy_input = self.pool1(self.conv1(dummy_input))
        dummy_input = self.pool2(self.conv2(dummy_input))
        dummy_input = self.pool3(self.conv3(dummy_input))

        return dummy_input.numel()

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
