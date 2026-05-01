import numpy as np
import torch
from torch.utils.data import Dataset


class ModularArithmeticDataset(Dataset):
    def __init__(self, P=97, seed=0, loss_criterion="MSE"):
        self.P = P
        self.seed = seed
        self.loss_criterion = loss_criterion

        # Instantiate the data
        self.data, self.indices = self.gen_data()

    def gen_data(self):
        data = torch.empty((self.P**2, 2 * self.P), dtype=torch.float32)
        if self.loss_criterion == "MSE":
            indices = torch.empty((self.P**2, self.P), dtype=torch.float32)
        else:
            indices = torch.empty((self.P**2, self.P), dtype=torch.long)
        for i in torch.arange(self.P):
            for j in torch.arange(self.P):
                combined_idx = self.P * i + j
                i_onehot = torch.nn.functional.one_hot(i, self.P)
                j_onehot = torch.nn.functional.one_hot(j, self.P)
                data[combined_idx] = torch.cat((i_onehot, j_onehot), axis=0)
                # indices[combined_idx] = (i+j) % self.P
                # if you wanted to implement MSE as MSE between the one-hots but better between the indices methinks.
                newval = (i + j) % (self.P)
                if self.loss_criterion == "MSE":
                    indices[combined_idx] = torch.nn.functional.one_hot(
                        newval, self.P
                    )
                    # I think to calculate it the other way around you, i.e. for MSE, you need
                    # indices[combined_idx]=newval
                else:
                    indices[combined_idx] = torch.nn.functional.one_hot(
                        newval, self.P
                    )
        return data, indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], self.indices[idx])
