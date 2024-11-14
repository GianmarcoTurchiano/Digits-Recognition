import argparse
import zipfile
import os

def unzip(zip_path, out_path):
    # Check if the specified path exists, create if it doesn't
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print(f"Created path: {out_path}")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(out_path)

    print(f"Files extracted to: {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-z', '--zip_path', type=str)
    parser.add_argument('-o', '--out_path', type=str)
    args = parser.parse_args()

    unzip(args.zip_path, args.out_path)