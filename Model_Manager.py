# Define the custom model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Union
# Define the custom model
class CustomModel(nn.Module):
    def __init__(self, input, input_size2, hidden_size, output_size):
        super(CustomModel, self).__init__()

        # Sub-network 1
        self.fc1 = nn.Linear(input, 16)

        # Sub-network 2
        self.fc2 = nn.Linear(input_size2, 16)

        # Combined network
        self.fc3 = nn.Linear(32, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x1, x2):
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)

        # Concatenate the output of sub-networks
        x = torch.cat((x1, x2), dim=-1)

        # Pass through the combined network
        x = self.fc3(x)
        x = self.fc4(x)

        return x

#picture
#filter
#concatonate output of filters