"""
Code for train, test, validation split of the dataset.
"""
import argparse

from sklearn.model_selection import train_test_split
from digits_recognition.experimentation.load_pickle_data import load_pickle_data
from digits_recognition.experimentation.dataset.save_pickle_data import save_pickle_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--train_set_path', type=str)
    parser.add_argument('--val_set_path', type=str)
    parser.add_argument('--validation_ratio', type=float)
    parser.add_argument('--split_seed', type=int)

    args = parser.parse_args()

    data = load_pickle_data(args.dataset_path)

    X_train, X_val, y_train, y_val = train_test_split(
        data['X'],
        data['y'],
        stratify=data['y'],
        test_size=args.validation_ratio,
        random_state=args.split_seed
    )

    save_pickle_data(args.train_set_path, X_train, y_train)
    save_pickle_data(args.val_set_path, X_val, y_val)

    # Check the shapes
    print("Training Images:", X_train.shape)
    print("Validation Images:", X_val.shape)
    print("Training Labels:", y_train.shape)
    print("Validation Labels:", y_val.shape)
