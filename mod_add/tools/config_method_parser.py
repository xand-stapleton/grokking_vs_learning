import torch
import torch.nn as nn

# Code to convert activation name into Torch method


def str_to_activation(activation_name):
    if activation_name.lower() == "relu":
        activation = nn.ReLU
    elif activation_name.lower() == "tanh":
        activation = nn.Tanh
    elif activation_name.lower() == "sigmoid":
        activation = nn.Sigmoid
    elif activation_name.lower() == "gelu":
        activation = nn.GELU
    else:
        raise NotImplementedError(
            "Activation function must be one of relu, gelu, tanh, or sigmoid"
        )
    return activation


def str_to_optimiser(optimiser_name):
    if optimiser_name.lower() == "adam":
        optimiser_fn = torch.optim.Adam  # 'Adam' or 'AdamW' or 'SGD'
    elif optimiser_name.lower == "adamw":
        optimiser_fn = torch.optim.AdamW  # 'Adam' or 'AdamW' or 'SGD'
    elif optimiser_name.lower == "sgd":
        optimiser_fn = torch.optim.SGD  # 'Adam' or 'AdamW' or 'SGD'
    else:
        raise NotImplementedError(
            f"Unrecognised optimiser name. Choose from adam, adamw, sgd. Name inputed: {optimiser_name}"
        )
    return optimiser_fn


def str_to_loss(loss_criterion):
    # set functions for neural network
    if loss_criterion.lower() == "crossentropy":
        loss_fn = nn.CrossEntropyLoss()  # 'MSELoss' or 'CrossEntropyLoss'
    else:
        loss_fn = nn.MSELoss()

    return loss_fn
