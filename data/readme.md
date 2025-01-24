# Dataset Card: MNIST (Modified National Institute of Standards and Technology Database)

## Table of Contents
1. [Dataset Description](#dataset-description)
2. [Domain Knowledge Overview](#domain-knowledge-overview)
3. [Folder Structure](#folder-structure)
4. [License and Permissions](#license-and-permissions)

## Dataset Description

The MNIST dataset is a widely used dataset in the field of computer vision, specifically for training and testing handwritten digit classification models. It consists of grayscale images (1 channel) of handwritten digits (0-9) and their corresponding labels. The dataset has been instrumental in the development of deep learning techniques and remains a popular choice for prototyping.

### Key Points:
- **Purpose:** Designed for academic research, model benchmarking, and algorithm development in machine learning and computer vision.
- **Usage Restrictions:** The dataset is publicly available and can be used for research and educational purposes.
- **Supplementary Documentation:** The dataset is further described in the original paper "Gradient-Based Learning Applied to Document Recognition" by LeCun et al.

## Domain Knowledge Overview

### MNIST Overview:
MNIST was created by Yann LeCun and colleagues by combining the digits from NIST’s Special Database 3 and Special Database 1. The dataset provides a standardized way to evaluate machine learning algorithms.

### Key Concepts in MNIST:
1. **Handwritten Digits:**
   - Each image is a 28x28 pixel grayscale image of a digit between 0 and 9.
   - Pixel values range from 0 (black) to 255 (white).

2. **Dataset Composition:**
   - **Training Set:** 60,000 images.
   - **Test Set:** 10,000 images.
   - The training and test sets have balanced representations of each digit class.

3. **Applications:**
   - Image classification tasks.
   - Evaluation of neural networks and computer vision models.
   - Research into unsupervised, supervised, and semi-supervised learning techniques.

## Folder Structure

### 1. Raw (`raw/`)
Contains the zipped dataset.

### 2. Interim (`interim/`)
The sub-folder `unzipped/` contains the raw image data split into training and test sets.

- **File Format:** The images are stored in IDX format, a simple binary file format.
- **Files:**
  - `train-images-idx3-ubyte`: Training set images.
  - `train-labels-idx1-ubyte`: Training set labels.
  - `t10k-images-idx3-ubyte`: Test set images.
  - `t10k-labels-idx1-ubyte`: Test set labels.

The `train_set.pkl` file is a pickle file containing the training set images (key `X`) and labels (key `y`), both converted into the numpy format.

### 3. Processed (`processed/`)
The `test_set.pkl` file is a pickle file containing the test set images (key `X`) and labels (key `y`), both converted into the numpy format.

The `train_set.pkl` and `val_set.pkl` files result from a split of the the `interim/train_set.pkl` file.

## License and Permissions

- **License:** The dataset is freely available under the Creative Commons Attribution-ShareAlike license.
- **Permissions:** Redistribution and usage are allowed for non-commercial purposes, provided appropriate credit is given to the authors.
- **Attribution:** 
  *Yann LeCun, Corinna Cortes, and Christopher J.C. Burges. "The MNIST Database of Handwritten Digits."*
