"""
Code for preparing the dataset for usage with PyTorch.
"""

import pickle

import torch
from torch.utils.data import DataLoader, TensorDataset


def _load_data(path):
    """
    Returns the content of a pickle file.
    """
    with open(path, 'rb') as file:
        data = pickle.load(file)

    return data


def load_data_tensors(path, normalize):
    """
    Returns images and labels columns from the 'X' and 'y' keys
    of a dictionary obtained from a file.
    """
    data = _load_data(path)

    images = torch.tensor(data['X'], dtype=torch.float32)
    labels = torch.tensor(data['y'], dtype=torch.long)

    if normalize:
        images = images / 255.0

    return images, labels


def load_dataset(path, normalize, shuffle, batch_size):
    """
    Returns a loader from the contents of a pickle file containing an 'X' and 'y' key.
    """
    images, labels = load_data_tensors(path, normalize)

    tensor_data = TensorDataset(images, labels)
    loader = DataLoader(tensor_data, batch_size=batch_size, shuffle=shuffle)

    return loader
