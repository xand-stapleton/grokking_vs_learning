import torch
from torch.utils.data import Dataset

from datasets.mod_add_dataset import ModularArithmeticDataset


class FisherDataset(Dataset):
    """
    Define the dataset for the Fisher metric evaluations

    The format of the dataset that the Fisher metric package NNGeometry needs
    is slightly weird. We want the data to be unlabelled (since we're
    evaluating the metric we don't care if the model prediction is right or
    wrong). This will be a subset of the training data, albeit in a slightly
    different form.
    """

    def __init__(self, test_dataset, num_samples):
        self.test_dataset = test_dataset
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (self.test_dataset[idx][0],)
