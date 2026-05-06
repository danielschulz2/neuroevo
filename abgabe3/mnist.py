# SPDX-FileCopyrightText: 2025 Timm Kleipsties <timm@kleipsties.tech>
#
# SPDX-License-Identifier: GPL-3.0-or-later

#########################################################
#                     Example Usage                     #
#########################################################
#                                                       #
# from mnist import *                                   #
#                                                       #
# check_and_download_mnist_files()                      #
#                                                       #
# X_train = load_mnist_images(TRAIN_IMAGES_PATH)        #
# y_train = load_mnist_labels(TRAIN_LABELS_PATH)        #
# X_test = load_mnist_images(TEST_IMAGES_PATH)          #
# y_test = load_mnist_labels(TEST_LABELS_PATH)          #
#                                                       #
# # if you need a one-hot-encoded representation        #
# y_train_one_hot = np.eye(10)[y_train]                 #
# y_test_one_hot = np.eye(10)[y_test]                   #
#                                                       #
# plot_mnist_example(X_test[0], y_test_one_hot[0])      #
#                                                       #
#########################################################


import gzip
import os
import shutil
import struct
import urllib.request
from array import array

import matplotlib.pyplot as plt
import numpy as np

TRAIN_IMAGES_PATH = (
    "mnist/train-images-idx3-ubyte.gz"
    if not os.path.exists("/kaggle/input")
    else "/kaggle/input/mnist-dataset/train-images.idx3-ubyte"
)
TRAIN_LABELS_PATH = (
    "mnist/train-labels-idx1-ubyte.gz"
    if not os.path.exists("/kaggle/input")
    else "/kaggle/input/mnist-dataset/train-labels.idx1-ubyte"
)
TEST_IMAGES_PATH = (
    "mnist/t10k-images-idx3-ubyte.gz"
    if not os.path.exists("/kaggle/input")
    else "/kaggle/input/mnist-dataset/t10k-images.idx3-ubyte"
)
TEST_LABELS_PATH = (
    "mnist/t10k-labels-idx1-ubyte.gz"
    if not os.path.exists("/kaggle/input")
    else "/kaggle/input/mnist-dataset/t10k-labels.idx1-ubyte"
)


def download_file(url: str, path: str) -> None:
    """
    Downloads a file from the given URL and saves it to the specified path.

    Args:
        url (str): The URL of the file to download.
        path (str): The path where the downloaded file will be saved.

    Returns:
        None
    """
    with urllib.request.urlopen(url) as response, open(path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)


def check_and_download_mnist_files() -> None:
    """
    Check if the MNIST files exist and download them if they don't.

    This function checks for the existence of MNIST files and downloads them from the specified URLs if they are not found.
    Also in case the folder "mnist" does not exist, it will be created.
    The files include the training images, training labels, test images, and test labels.

    Args:
        None

    Returns:
        None
    """
    global TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, TEST_IMAGES_PATH, TEST_LABELS_PATH
    files = [
        (
            "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
            TRAIN_IMAGES_PATH,
        ),
        (
            "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
            TRAIN_LABELS_PATH,
        ),
        (
            "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
            TEST_IMAGES_PATH,
        ),
        (
            "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
            TEST_LABELS_PATH,
        ),
    ]

    if os.path.exists("/kaggle/input") and os.path.exists(
        "/kaggle/input/mnist-dataset"
    ):
        print(
            "You seem to use Kaggle, but have not downloaded the mnist dataset yet. Please search for input 'MNIST Dataset' by Hojjat Khodabakhsh (23MB)"
        )

    if not os.path.exists("mnist") and not os.path.exists("/kaggle/input"):
        os.makedirs("mnist")

    for url, filename in files:
        if not os.path.exists(filename):
            print(f"File {filename} not found, start downloading...")
            try:
                download_file(url, filename)
            except Exception as e:
                print(
                    f"An error occurred while downloading {filename}. Maybe try again downloading the file."
                )
                print(e)
        else:
            print(f"File {filename} already exists.")


def read_images_labels(images_filepath, labels_filepath):
    """
    Helper method for using Kaggle and the MNIST Dataset by Hojjat Khodabakhsh (23MB)
    """
    labels = []

    if labels_filepath is not None:
        with open(labels_filepath, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(
                    "Magic number mismatch, expected 2049, got {}".format(magic)
                )
            labels = array("B", file.read())

    images = []

    if images_filepath is not None:
        with open(images_filepath, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(
                    "Magic number mismatch, expected 2051, got {}".format(magic)
                )
            image_data = array("B", file.read())
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols : (i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

    return images, labels


def load_mnist_images(filename: str) -> np.ndarray:
    """
    Load MNIST images from a file and flattens and normalizes the data.

    Args:
        filename (str): The path to the MNIST image file.

    Returns:
        numpy.ndarray: An array of shape (num_images, 784) containing the normalized MNIST images.
    """
    if os.path.exists("/kaggle/input"):
        data = np.array(read_images_labels(filename, None)[0])
    else:
        with gzip.open(filename, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 784)  # 28x28 Pictues flattend to 784
    return data / np.float32(255)  # normalize


def load_mnist_labels(filename: str) -> np.ndarray:
    """
    Load MNIST labels from a file.

    Parameters:
    filename (str): The path to the file containing the MNIST labels.

    Returns:
    numpy.ndarray: An array of MNIST labels.

    """
    if os.path.exists("/kaggle/input"):
        data = np.array(read_images_labels(None, filename)[1])
    else:
        with gzip.open(filename, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


def plot_mnist_example(X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Visualizes a random test example from the given dataset.

    Parameters:
    - X_test (numpy.ndarray): Array of input test examples.
    - y_test (numpy.ndarray): Array of corresponding labels for the test examples.

    Returns:
    None
    """
    image = X_test.reshape(28, 28)
    label = y_test

    plt.imshow(image, cmap="gray")
    plt.title(f"Label: {np.argmax(label)} = {label}")
    plt.axis("off")
    plt.show()
