"""
Code for preparing the dataset for usage with PyTorch.
"""

import pickle

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np

from digits_recognition.modeling.classifier import INPUT_HEIGHT, INPUT_WIDTH


def _load_data(path):
    """
    Returns the content of a pickle file.
    """
    with open(path, 'rb') as file:
        data = pickle.load(file)

    return data


class DigitsDataset(Dataset):
    """
    Returns image-label pairs and applies transformations to images.
    """
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = image.astype(np.uint8)
        image = Image.fromarray(image, mode='L')

        if self.transform:
            image = self.transform(image)

        image = F.to_tensor(image)
        label = torch.tensor(label)

        return image, label


rotation_transform = transforms.RandomRotation(degrees=90)
gaussian_blur_transform = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))
resize_crop_transform = transforms.RandomResizedCrop(
    size=(INPUT_HEIGHT, INPUT_WIDTH),
    scale=(0.8, 1.0)
)
data_augmentation = transforms.Compose([
    rotation_transform,
    gaussian_blur_transform,
    resize_crop_transform,
])


def load_dataset(path, shuffle, batch_size, augment=False):
    """
    Returns a loader from the contents of a pickle file containing an 'X' and 'y' key.
    """
    data = _load_data(path)

    if augment:
        transform = data_augmentation
    else:
        transform = None

    dataset = DigitsDataset(data['X'], data['y'], transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader
