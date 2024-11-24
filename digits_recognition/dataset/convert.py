"""
Code for train, test, validation split of the dataset.
"""
import argparse

from digits_recognition.save_pickle_data import save_pickle_data
from digits_recognition.dataset.load_ubyte_data import load_ubyte_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--in_path', type=str)
    parser.add_argument('-tr', '--train_set_path', type=str)
    parser.add_argument('-ts', '--test_set_path', type=str)

    args = parser.parse_args()

    # Load the images and labels
    train_images, train_labels, test_images, test_labels = load_ubyte_data(args.in_path)

    save_pickle_data(args.train_set_path, train_images, train_labels)
    save_pickle_data(args.test_set_path, test_images, test_labels)

    # Check the shapes
    print("Training Images:", train_labels.shape)
    print("Training Labels:", train_labels.shape)
    print("Test Images:", test_images.shape)
    print("Test Labels:", test_labels.shape)
