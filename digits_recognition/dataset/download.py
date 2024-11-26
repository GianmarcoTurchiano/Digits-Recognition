"""
Script for downloading and saving the raw dataset.
"""
import argparse
import sys

import requests


DATASET_URL = r'https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset'


def download(url):
    """
    Downloads the raw dataset.
    """

    return requests.get(url, allow_redirects=True, timeout=1000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--out_path', type=str)
    args = parser.parse_args()

    response = download(DATASET_URL)

    if response.status_code != 200:
        print(f'Request to {DATASET_URL} returned {response.status_code}')
        sys.exit(1)

    with open(args.out_path, 'wb') as file:
        file.write(response.content)

    print(f"Files downloaded to: {args.out_path}")
