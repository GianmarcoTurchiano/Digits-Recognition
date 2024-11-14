from digits_recognition.dataset import load_ubyte_images, load_ubyte_labels, save_dataset
import argparse
from sklearn.model_selection import train_test_split
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d', '--in_path', type=str)
    parser.add_argument('-o', '--out_path', type=str)
    parser.add_argument('-r', '--validation_ratio', type=float)
    parser.add_argument('-s', '--split_seed', type=int)

    args = parser.parse_args()

    # Load the images and labels
    train_images = load_ubyte_images(f'{args.in_path}/train-images-idx3-ubyte/train-images-idx3-ubyte')
    train_labels = load_ubyte_labels(f'{args.in_path}/train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images = load_ubyte_images(f'{args.in_path}/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels = load_ubyte_labels(f'{args.in_path}/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    # Split the data: 80% for training, 20% for validation
    X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, stratify=train_labels,
                                                      test_size=args.validation_ratio, random_state=args.split_seed)

    save_dataset(args.out_path, X_train, y_train, X_val, y_val, test_images, test_labels)

    # Check the shapes
    print("Training Images:", X_train.shape)
    print("Validation Images:", X_val.shape)
    print("Training Labels:", y_train.shape)
    print("Validation Labels:", y_val.shape)
    print("Test Images:", test_images.shape)
    print("Test Labels:", test_labels.shape)