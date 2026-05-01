import pickle
import random

import numpy as np
import torch
from torch.utils.data import Dataset


def create_ising_dataset(
    data_seed, train_size, test_size, cluster=True, dtype=torch.float32
):
    random.seed(data_seed)
    for set_seed in [
        torch.manual_seed,
        torch.cuda.manual_seed_all,
        np.random.seed,
    ]:
        set_seed(data_seed)

    if cluster:
        datafilename = "../Data/IsingML_L16_traintest.pickle"
    else:
        datafilename = "/Users/dmitrymanning-coe/Documents/Research/Grokking/Ising_Code/Data/IsingML_L16_traintest.pickle"

    with open(datafilename, "rb") as handle:
        data = pickle.load(handle)
        print(f"Data length: {len(data[1])}")
        print(data[0])
        data = data[1]
    # shuffle data list
    random.shuffle(data)
    # split data into input (array) and labels (phase and temp)
    inputs, phase_labels, temp_labels = zip(*data)
    # for now ignore temp labels
    my_X = torch.Tensor(np.array(inputs)).to(
        dtype
    )  # transform to torch tensor of FLOATS
    my_y = torch.Tensor(np.array(phase_labels)).to(
        torch.long
    )  # transform to torch tensor of INTEGERS
    my_y_temp = torch.Tensor(np.array(temp_labels)).to(dtype)
    print(my_X.dtype, my_y.dtype)
    print("Created Ising Dataset")

    # manually do split between training and testing data (only necessary if ising data)
    # otherwise use torch.utils.data.Subset to get subset of MNIST data
    train_size, test_size, batch_size = train_size, test_size, train_size
    a, b = train_size, test_size
    train_data = TensorDataset(
        my_X[b : a + b], my_y[b : a + b]
    )  # Choose training data of specified size
    test_data = TensorDataset(my_X[:b], my_y[:b])  # test
    scramble_snapshot = False

    # load data in batches for reduced memory usage in learning
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=test_size, shuffle=False)
    for b, (X_train, y_train) in enumerate(train_loader):
        print("batch:", b)
        print("input tensors shape (data): ", X_train.shape)
        print("output tensors shape (labels): ", y_train.shape)

    for b, (X_test, y_test) in enumerate(test_loader):
        if scramble_snapshot:
            X_test_a = np.array(X_test)
            X_test_perc = np.zeros((test_size, 16, 16))
            for t in range(test_size):
                preshuff = X_test_a[t, :, :].flatten()
                np.random.shuffle(preshuff)
                X_test_perc[t, :, :] = np.reshape(preshuff, (16, 16))
            X_test = torch.Tensor(X_test_perc).to(dtype)
        print("batch:", b)
        print("input tensors shape (data): ", X_test.shape)

        print("output tensors shape (labels): ", y_test.shape)

    # desc='data[1] is [[data_seed,sgd_seed,init_seed,np.random.randint(1,1000,1000)],[X_test,y_test],[X_train,y_train]]'
    # data_save1=[[data_seed,np.random.randint(1,1000,1000)],[X_test,y_test],[X_train,y_train]]
    # data_save=[desc,data_save1]
    return train_loader, test_loader


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
