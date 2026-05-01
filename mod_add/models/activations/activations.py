import torch
import torch.nn as nn


class Square(nn.Module):
    """
    Torch-friendly implementation of activation if one wants to use
    quadratic activations a la Gromov (induces faster grokking).
    """

    def forward(self, x):
        return torch.square(x)
