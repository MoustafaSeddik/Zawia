#%% import necessary libraries
import torch
import torch.nn as nn
from colorama import Fore, Style
from torchsummary import summary
from src.settings.settings import device

#%% Define the LinearModel class
class LinearModel(nn.Module):
    def __init__(self, input_size, hidden_units, hidden_layer, width_type, dropout_rate):
        super(LinearModel, self).__init__()
        units = []
        if width_type == 'constant':
            # Creates [h, h, h, h]
            units = [hidden_units] * int(hidden_layer)
        elif width_type == 'upward':
            # Creates [h, 2h, 4h, 8h]
            units = [min(2048, hidden_units * (2 ** i)) for i in range(int(hidden_layer))]
        elif width_type == 'downward':
            # Creates [h, h/2, h/4, h/8]
            units = [max(64, hidden_units // (2 ** i)) for i in range(int(hidden_layer))]

        print(f"{Fore.RED}hidden_units:{Style.RESET_ALL}{units}")

        layers = []
        # First layer
        layers.append(nn.Linear(input_size, units[0]))
        #layers.append(nn.BatchNorm1d(units[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_rate, inplace=False))

        # Hidden layers
        for i in range(hidden_layer - 1):
            layers.append(nn.Linear(units[i], units[i + 1]))
            #layers.append(nn.BatchNorm1d(units[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate, inplace=False))

        # Output layer
        layers.append(nn.Linear(units[-1], 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def summary(self, input_size):
        print(f"{Fore.RED}Model Summary:{Style.RESET_ALL}")
        # Normalize the device to the string format expected by torchsummary
        if device is None:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        elif isinstance(device, torch.device):
            device_str = device.type  # "cpu" or "cuda"
        else:
            device_str = str(device).lower()

        return summary(self.model, input_size=input_size,
                       device=device_str)

    def get_modelParameters(self):
        model_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        #print(f"{Fore.RED}Total trainable parameters:{Style.RESET_ALL} {model_params:,}")
        return model_params

