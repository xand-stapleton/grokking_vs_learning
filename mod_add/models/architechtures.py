import os
import sys

# Add the path to the PYTHONPATH - This was from before I added the __init__.py file to the mod_add folder
# new_path = "/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/Code"
# if new_path not in sys.path:
#     sys.path.append(new_path)
#     os.environ["PYTHONPATH"] = os.pathsep.join(sys.path)
import torch
import torch.nn as nn

from tools.wrappers import *


class CNN(nn.Module):
    @store_init_args
    def __init__(
        self,
        input_dim,
        output_size,
        input_channels,
        conv_channels,
        hidden_widths,
        activation=nn.ReLU(),
        optimizer=torch.optim.Adam,
        learning_rate=0.001,
        weight_decay=0,
        multiplier=1,
        dropout_prob=0.01,
    ):
        super().__init__()

        # transforms input from input_channels x input_dim x input_dim
        # to out_channels x input_dim x input_dim
        conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=conv_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        conv2 = nn.Conv2d(
            in_channels=conv_channels[0],
            out_channels=conv_channels[1],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        # divide shape of input_dim by two after applying each convolution
        # since this is applied twice, make sure that input_dim is divisible by 4!!!
        pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Construct convolution and pool layers
        self.conv_layers = nn.ModuleList()
        for conv_layer in [conv1, conv2]:
            self.conv_layers.append(conv_layer)
            self.conv_layers.append(activation)
            self.conv_layers.append(pool)

        # dropout to apply in FC layers
        self.dropout = nn.Dropout(dropout_prob)

        # construct fully connected layers
        self.fc_layers = nn.ModuleList()
        # flattened size after two convolutions and two poolings of data
        input_size = (input_dim // 4) ** 2 * conv_channels[1]
        for size in hidden_widths:
            self.fc_layers.append(nn.Linear(input_size, size))
            input_size = size  # For the next layer
            self.fc_layers.append(activation)
            self.fc_layers.append(self.dropout)
        # add last layer without activation or dropout
        self.fc_layers.append(nn.Linear(input_size, output_size))

        # multiply weights by overall factor
        if multiplier != 1:
            with torch.no_grad():
                for param in self.parameters():
                    param.data = multiplier * param.data

        # use GPU if available
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

        # set optimizer
        self.optimizer = optimizer(
            params=self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def forward(self, X):
        # convolution and pooling
        for layer in self.conv_layers:
            X = layer(X)
        # fully connected layers
        X = X.view(X.size(0), -1)  # Flatten data for FC layers
        for layer in self.fc_layers:
            X = layer(X)
        return X


class CNN_nobias(nn.Module):
    def __init__(
        self,
        input_dim,
        output_size,
        input_channels,
        conv_channels,
        hidden_widths,
        activation=nn.ReLU(),
        optimizer=torch.optim.SGD,
        learning_rate=0.001,
        weight_decay=0,
        multiplier=1,
        dropout_prob=0.01,
    ):
        super().__init__()

        # transforms input from input_channels x input_dim x input_dim
        # to out_channels x input_dim x input_dim
        conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=conv_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        conv2 = nn.Conv2d(
            in_channels=conv_channels[0],
            out_channels=conv_channels[1],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        # divide shape of input_dim by two after applying each convolution
        # since this is applied twice, make sure that input_dim is divisible by 4!!!
        pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Construct convolution and pool layers
        self.conv_layers = nn.ModuleList()
        for conv_layer in [conv1, conv2]:
            self.conv_layers.append(conv_layer)
            self.conv_layers.append(activation)
            self.conv_layers.append(pool)

        # dropout to apply in FC layers
        self.dropout = nn.Dropout(dropout_prob)

        # construct fully connected layers
        self.fc_layers = nn.ModuleList()
        # flattened size after two convolutions and two poolings of data
        input_size = (input_dim // 4) ** 2 * conv_channels[1]
        for size in hidden_widths:
            self.fc_layers.append(nn.Linear(input_size, size, bias=False))
            input_size = size  # For the next layer
            self.fc_layers.append(activation)
            self.fc_layers.append(self.dropout)
        # add last layer without activation or dropout
        self.fc_layers.append(nn.Linear(input_size, output_size, bias=False))

        # multiply weights by overall factor
        if multiplier != 1:
            with torch.no_grad():
                for param in self.parameters():
                    param.data = multiplier * param.data

        # use GPU if available
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

        # set optimizer
        self.optimizer = optimizer(
            params=self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def forward(self, X):
        # convolution and pooling
        for layer in self.conv_layers:
            X = layer(X)
        # fully connected layers
        X = X.view(X.size(0), -1)  # Flatten data for FC layers
        for layer in self.fc_layers:
            X = layer(X)
        return X


class MLP(nn.Module):
    @store_init_args
    def __init__(
        self,
        hidden=[512],
        P=97,
        weight_multiplier=1.0,
        bias=True,
    ):
        super(MLP, self).__init__()
        layers = []
        input_dim = 2 * P
        first = True
        for layer_ind in range(len(hidden)):
            if first:
                layers.append(
                    nn.Linear(input_dim, hidden[layer_ind], bias=bias)
                )
                layers.append(nn.ReLU())
                first = False
            else:
                layers.append(
                    nn.Linear(
                        hidden[layer_ind - 1], hidden[layer_ind], bias=bias
                    )
                )
                layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden[-1], P, bias=bias))
        self.model = nn.Sequential(*layers)

        # I'm going to define the optimizer within the AnalysableModel class instead so that we
        # don't have two optimizers within that class.
        # self.optimizer = optimizer(
        #     params=self.parameters(), lr=learning_rate, weight_decay=weight_decay
        # )
        self.init_weights()

        with torch.no_grad():
            for param in self.parameters():
                param.data = weight_multiplier * param.data

    def forward(self, x):
        x = self.model(x)
        return x

    # Consider getting rid of the xavier normal init
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
