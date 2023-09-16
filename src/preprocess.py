import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import Config


# ricord, rsna, cohen, actmed, sirm,
def getClass(label):
    if label == "negative":
        return "NORMAL"
    if label == "positive":
        return "COVID"


def create_image_dataframe(txt_path, dataset_type):
    """
    Create a pandas DataFrame containing image file paths and labels based on information from a text file.

    Args:
        txt_path (str): The path to the text file containing image information.
        dataset_type (str): The type of dataset ('train', 'test', or 'val').

    Returns:
        pandas.DataFrame: A DataFrame containing image file paths and corresponding labels.

    """
    txt_file = open(txt_path, "r")
    lines = txt_file.readlines()
    image_data = []

    image_formats = ["jpg", "jpeg", "png"]

    for line in lines:
        words = line.split()

        if len(words) >= 4:
            image_id = words[0]
            image_path = Config.DataDir + "/" + dataset_type + "/" + words[1]
            label = words[2]

            for img_format in image_formats:
                if img_format in line:
                    image_data.append(
                        {"Filepath": image_path, "Label": getClass(label)}
                    )
                    break

    image_df = pd.DataFrame(image_data)
    return image_df


def print_images(samples, save_path=None):
    """
    Display or save a figure containing a grid of images.

    Args:
        samples (DataFrame): DataFrame containing image filepaths and labels.
        save_path (str, optional): Path to save the figure. If None, the figure will be displayed instead.

    Returns:
        None
    """
    images = samples["Filepath"].to_numpy()
    labels = samples["Label"].to_numpy()

    fig = plt.figure(figsize=(20, 8))
    columns = 4
    rows = 1

    for i, image_path in enumerate(images):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        fig.add_subplot(rows, columns, i + 1)
        title = "{}".format(labels[i])

        Sample_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)

        plt.imshow(Sample_image, cmap="gray")
        plt.title(title)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def create_gen(frame_, batch_size, seed, IMG_SIZE):
    """
    Create a generator to preprocess and augment images from a DataFrame.

    Args:
        frame_ (DataFrame): DataFrame containing image filepaths and labels.
        batch_size (int): Number of samples per batch.
        seed (int): Random seed for reproducibility.
        IMG_SIZE (tuple): Tuple specifying the target image size (height, width).

    Returns:
        generator (ImageDataGenerator): ImageDataGenerator object configured with the provided parameters.
    """

    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        brightness_range=[1.0, 1.3],
        rotation_range=15,
        # zoom_range=0.2
    )
    generator = datagen.flow_from_dataframe(
        dataframe=frame_,
        x_col="Filepath",
        y_col="Label",
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
        class_mode="sparse",
        color_mode="rgb",
        save_format="jpeg",
        target_size=IMG_SIZE,
    )

    return generator


# Example usage:
# generator = create_gen(frame_, batch_size=32, seed=42, IMG_SIZE=(224, 224))
