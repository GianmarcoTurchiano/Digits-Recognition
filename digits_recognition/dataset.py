import requests
import zipfile
import os
import numpy as np
import pickle

def download(url, out_path):
    response = requests.get(url, allow_redirects=True)

    with open(out_path, 'wb') as file:
        file.write(response.content)

    print(f"Files downloaded to: {out_path}")

def unzip(zip_path, out_path):
    # Check if the specified path exists, create if it doesn't
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print(f"Created path: {out_path}")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(out_path)

    print(f"Files extracted to: {out_path}")

def load_ubyte_images(filename):
    with open(filename, 'rb') as f:
        # Skip the header (first 16 bytes for images)
        f.read(16)
        # Read the rest as a numpy array, reshape to 28x28 per image
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
    return data

def load_ubyte_labels(filename):
    with open(filename, 'rb') as f:
        # Skip the header (first 8 bytes for labels)
        f.read(8)
        # Read the rest as a numpy array
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def save_dataset(out_path, X_train, y_train, X_val, y_val, X_test, y_test):
    with open(out_path, 'wb') as file:
        pickle.dump({
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }, file)

def load_dataset(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data
