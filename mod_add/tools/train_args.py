from dataclasses import dataclass
from typing import List, Union

import torch.nn as nn


@dataclass
class TrainArgs:
    epochs: int
    lr: float
    weight_decay: float
    weight_multiplier: float
    dropout_prob: float
    data_seed: int | None = 0
    sgd_seed: int | None = 0
    init_seed: int | None = 0
    device: str = "cpu"
    test_size: int = 1000
    train_size: int = 100
    hiddenlayers: list | None = None
    conv_channels: list | None = None
    train_fraction: float = 0.5
    P: int = 97
    loss_criterion: str = "crossentropy"
    batch_size: int = 64
    activation: any = nn.ReLU
    fisher_bs: int = 16
    fisher_seed: int | None = 0
    root_dir: str = ""
