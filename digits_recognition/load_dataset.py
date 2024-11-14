import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset

def _load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data

def load_dataset(path, normalize, shuffle, batch_size):
    data = _load_data(path)

    X = torch.tensor(data['X'], dtype=torch.float32)
    y = torch.tensor(data['y'], dtype=torch.long)

    if normalize:
        X = X / 255.0

    tensor_data = TensorDataset(X, y)
    loader = DataLoader(tensor_data, batch_size=batch_size, shuffle=shuffle)

    return loader