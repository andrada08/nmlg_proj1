import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_size, hidden_sizes: tuple[int, int, int], output_size: int, activation: str = "relu"):
        super(Net, self).__init__()
        self.input_size = input_size
        h1, h2, h3 = hidden_sizes
        self.layer1 = nn.Linear(input_size, h1)
        self.layer2 = nn.Linear(h1, h2)
        # Layer3 receives inputs from both layer1 and layer2
        self.layer3_from_1 = nn.Linear(h1, h3)  # W13: layer1→layer3
        self.layer3_from_2 = nn.Linear(h2, h3)  # W23: layer2→layer3
        self.layer3 = nn.Linear(h3, output_size)  # Layer3 directly outputs to final size
        # choose activation by name (strict: will raise if not found)
        self.activation = getattr(F, activation)

    def forward(self, x):
        x = x.view(-1, self.input_size)

        x1 = self.activation(self.layer1(x))
        x2 = self.activation(self.layer2(x1))
        
        # Layer3 combines inputs from both layer1 and layer2
        x3_combined = self.layer3_from_1(x1) + self.layer3_from_2(x2)
        output = self.layer3(x3_combined)  # Layer3 directly produces final output
        return output

    def get_layer_names(self):
        return ["layer1", "layer2", "layer3_from_1", "layer3_from_2", "layer3"]