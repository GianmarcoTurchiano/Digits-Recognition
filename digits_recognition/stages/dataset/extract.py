from digits_recognition.dataset import unzip
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d', '--zip_path', type=str)
    parser.add_argument('-o', '--out_path', type=str)
    args = parser.parse_args()

    unzip(args.zip_path, args.out_path)