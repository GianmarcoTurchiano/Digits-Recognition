import torch
import torch.nn as nn

INPUT_WIDTH = 28
INPUT_HEIGHT = 28
CHANNEL_AMOUNT = 1
CLASS_AMOUNT = 10

class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(INPUT_HEIGHT * INPUT_HEIGHT, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, CLASS_AMOUNT)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x