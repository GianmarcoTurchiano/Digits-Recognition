"""
Code for reading ubyte formatted data as numpy arrays.
"""
import numpy as np


TRAIN_IMAGES_PATH = 'train-images-idx3-ubyte/train-images-idx3-ubyte'
TRAIN_LABELS_PATH = 'train-labels-idx1-ubyte/train-labels-idx1-ubyte'
TEST_IMAGES_PATH = 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
TEST_LABELS_PATH = 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'


def _load_ubyte_images(filename):
    """
    Returns the content of a ubyte file parsed as int matrices.
    """
    with open(filename, 'rb') as f:
        # Skip the header (first 16 bytes for images)
        f.read(16)
        # Read the rest as a numpy array, reshape to 28x28 per image
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
    return data


def _load_ubyte_labels(filename):
    """
    Returns the content of a ubyte file parsed as int scalars.
    """
    with open(filename, 'rb') as f:
        # Skip the header (first 8 bytes for labels)
        f.read(8)
        # Read the rest as a numpy array
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def load_ubyte_data(path):
    """
    Reads train set and test set from ubyte data and then returns them.
    """
    train_images = _load_ubyte_images(f'{path}/{TRAIN_IMAGES_PATH}')
    test_images = _load_ubyte_images(f'{path}/{TEST_IMAGES_PATH}')
    train_labels = _load_ubyte_labels(f'{path}/{TRAIN_LABELS_PATH}')
    test_labels = _load_ubyte_labels(f'{path}/{TEST_LABELS_PATH}')

    return train_images, train_labels, test_images, test_labels
