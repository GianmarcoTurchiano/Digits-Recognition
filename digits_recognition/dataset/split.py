"""
Code for train, test, validation split of the dataset.
"""
import argparse
import pickle

import numpy as np
from sklearn.model_selection import train_test_split


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


def save_dataset(out_path, features, labels):
    """
    Saves a pickle file with keys 'X' and 'y'.
    """
    with open(out_path, 'wb') as file:
        pickle.dump({
            'X': features,
            'y': labels
        }, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--in_path', type=str)
    parser.add_argument('-tr', '--train_set_path', type=str)
    parser.add_argument('-ts', '--test_set_path', type=str)
    parser.add_argument('-v', '--val_set_path', type=str)
    parser.add_argument('-r', '--validation_ratio', type=float)
    parser.add_argument('-s', '--split_seed', type=int)

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

    # Split the data: 80% for training, 20% for validation
    X_train, X_val, y_train, y_val = train_test_split(
        train_images,
        train_labels,
        stratify=train_labels,
        test_size=args.validation_ratio,
        random_state=args.split_seed
    )

    save_dataset(args.train_set_path, X_train, y_train)
    save_dataset(args.val_set_path, X_val, y_val)
    save_dataset(args.test_set_path, test_images, test_labels)

    # Check the shapes
    print("Training Images:", X_train.shape)
    print("Validation Images:", X_val.shape)
    print("Training Labels:", y_train.shape)
    print("Validation Labels:", y_val.shape)
    print("Test Images:", test_images.shape)
    print("Test Labels:", test_labels.shape)
