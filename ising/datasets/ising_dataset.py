import pickle
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


def create_ising_dataset(
    data_seed,
    train_size,
    test_size,
    batch_size=None,
    cluster=True,
    dtype=torch.float32,
    scramble_snapshot=False,
    test_shuffle=False,
):
    if batch_size is None:
        batch_size = train_size

    random.seed(data_seed)
    for set_seed in [
        torch.manual_seed,
        torch.cuda.manual_seed_all,
        np.random.seed,
    ]:
        set_seed(data_seed)

    datafilename = "datasets/blobs/IsingML_L16_traintest.pickle"

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
    my_X = torch.Tensor(np.array(inputs)).to(dtype)
    # Need to add a channel dimension to the data
    my_X = my_X.reshape(-1, 1, 16, 16)

    # transform to torch tensor of FLOATS
    my_y = torch.Tensor(np.array(phase_labels)).to(
        torch.long
    )  # transform to torch tensor of INTEGERS
    # for consistency with modadd, and to make MSE loss run the same way as crossentropy
    my_y_onehot = torch.nn.functional.one_hot(my_y, num_classes=2)

    my_y_temp = torch.Tensor(np.array(temp_labels)).to(dtype)
    print(my_X.dtype, my_y.dtype)
    print("Created Ising Dataset")

    # manually do split between training and testing data (only necessary if ising data)
    # otherwise use torch.utils.data.Subset to get subset of MNIST data
    # train_size, test_size, batch_size = train_size, test_size, train_size
    a, b = train_size, test_size
    train_data = TensorDataset(
        my_X[b : a + b], my_y_onehot[b : a + b]
    )  # Choose training data of specified size
    test_data = TensorDataset(my_X[:b], my_y_onehot[:b])  # test

    # load data in batches for reduced memory usage in learning
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_data, batch_size=test_size, shuffle=test_shuffle
    )

    # for b, (X_train, y_train) in enumerate(train_loader):
    #     print("batch:", b)
    #     print("input tensors shape (data): ", X_train.shape)
    #     print("output tensors shape (labels): ", y_train.shape)

    if scramble_snapshot:
        for b, (X_test, y_test) in enumerate(test_loader):
            X_test_a = np.array(X_test)
            X_test_perc = np.zeros((test_size, 1, 16, 16))
            for t in range(test_size):
                preshuff = X_test_a[t, :, :, :].flatten()
                np.random.shuffle(preshuff)
                X_test_perc[t, :, :, :] = np.reshape(preshuff, 1, (16, 16))
            X_test = torch.Tensor(X_test_perc).to(dtype)

    # shape information on the data sizes, to cheer the weary debugger
    b, (X_test, y_test) = next(enumerate(test_loader))
    print("first batch:", b)
    print("input tensors shape (data): ", X_test.shape)

    print("output tensors shape (labels): ", y_test.shape)

    # desc='data[1] is [[data_seed,sgd_seed,init_seed,np.random.randint(1,1000,1000)],[X_test,y_test],[X_train,y_train]]'
    # data_save1=[[data_seed,np.random.randint(1,1000,1000)],[X_test,y_test],[X_train,y_train]]
    # data_save=[desc,data_save1]
    return train_loader, test_loader
