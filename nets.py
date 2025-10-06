import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_size, hidden_sizes: tuple[int, int], output_size: int):
        super(Net, self).__init__()
        self.input_size = input_size
        h1, h2 = hidden_sizes
        self.layer1 = nn.Linear(input_size, h1)
        self.layer2 = nn.Linear(h1, h2)
        self.layer3 = nn.Linear(h1 + h2, output_size)

    def forward(self, x):
        x = x.view(-1, self.input_size)

        x1 = F.relu(self.layer1(x))
        x2 = F.relu(self.layer2(x1))
        # Concatenate and output
        x12 = torch.cat([x1, x2], dim=1)
        x3 = self.layer3(x12)
        return x3

    def get_layer_names(self):
        return ["layer1", "layer2", "layer3"]