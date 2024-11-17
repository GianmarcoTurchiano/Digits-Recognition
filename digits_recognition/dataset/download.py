"""
Script for downloading the raw dataset.
"""
import argparse

import requests

DATASET_URL = r'https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset'


def download(url, out_path):
    """
    Downloads the raw dataset.
    """
    response = requests.get(url, allow_redirects=True, timeout=1000)

    with open(out_path, 'wb') as file:
        file.write(response.content)

    print(f"Files downloaded to: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--out_path', type=str)
    args = parser.parse_args()

    download(DATASET_URL, args.out_path)
