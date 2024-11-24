"""
Code for train, test, validation split of the dataset.
"""
import argparse

import numpy as np

from digits_recognition.save_pickle_data import save_pickle_data


def load_ubyte_images(filename):
    """
    Returns the content of a ubyte file parsed as int matrices.
    """
    with open(filename, 'rb') as f:
        # Skip the header (first 16 bytes for images)
        f.read(16)
        # Read the rest as a numpy array, reshape to 28x28 per image
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
    return data


def load_ubyte_labels(filename):
    """
    Returns the content of a ubyte file parsed as int scalars.
    """
    with open(filename, 'rb') as f:
        # Skip the header (first 8 bytes for labels)
        f.read(8)
        # Read the rest as a numpy array
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--in_path', type=str)
    parser.add_argument('-tr', '--train_set_path', type=str)
    parser.add_argument('-ts', '--test_set_path', type=str)

    args = parser.parse_args()

    # Load the images and labels
    train_images = load_ubyte_images(
        f'{args.in_path}/train-images-idx3-ubyte/train-images-idx3-ubyte'
    )
    train_labels = load_ubyte_labels(
        f'{args.in_path}/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
    )
    test_images = load_ubyte_images(
        f'{args.in_path}/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
    )
    test_labels = load_ubyte_labels(
        f'{args.in_path}/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
    )

    save_pickle_data(args.train_set_path, train_images, train_labels)
    save_pickle_data(args.test_set_path, test_images, test_labels)

    # Check the shapes
    print("Training Images:", train_labels.shape)
    print("Training Labels:", train_labels.shape)
    print("Test Images:", test_images.shape)
    print("Test Labels:", test_labels.shape)
