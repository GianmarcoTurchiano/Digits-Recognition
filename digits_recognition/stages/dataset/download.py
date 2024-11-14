from digits_recognition.dataset import download
from digits_recognition.config import DATASET_URL
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-o', '--out_path', type=str)
    args = parser.parse_args()

    download(DATASET_URL, args.out_path)