"""
Code for preparing the dataset for usage with PyTorch.
"""
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np

from digits_recognition.experimentation.pickle_data import load_pickle_data


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


rotation_transform = transforms.RandomRotation(degrees=20)
gaussian_blur_transform = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))
data_augmentation = transforms.Compose([
    rotation_transform,
    gaussian_blur_transform,
])


def get_data_loader(
    path,
    batch_size,
    device,
    num_workers=0,
    persistent_workers=False,
    augment=False,
    shuffle=False
):
    """
    Returns a loader from the contents of a pickle file containing an 'X' and 'y' key.
    """
    data = load_pickle_data(path)

    if augment:
        transform = data_augmentation
    else:
        transform = None

    dataset = DigitsDataset(data['X'], data['y'], transform=transform)

    pin_memory = device.type == 'cuda'

    if pin_memory:
        pin_memory_device = device.type
    else:
        pin_memory_device = ''

    if num_workers > 0:
        prefetch_factor = 8
    else:
        prefetch_factor = None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        pin_memory_device=pin_memory_device,
        prefetch_factor=prefetch_factor
    )

    return loader
