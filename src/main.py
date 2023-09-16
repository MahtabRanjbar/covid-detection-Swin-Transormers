import os
import random

import numpy as np
import pandas as pd
from keras import regularizers
from sklearn.model_selection import train_test_split

from config import Config
from evaluation import (
    display_confusion_matrix,
    evaluate_model,
    plot_training_history,
    predict_label,
    save_classification_report,
)
from preprocess import create_gen, create_image_dataframe, print_images
from train import BuildModel, FitModel, getCallbacks


def main():
    seed = 73
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = "73"

    # Load and preprocess the image data
    train_map = create_image_dataframe(Config.DataDir + "/train.txt", "train").sample(
        frac=1, random_state=73
    )
    test_map = create_image_dataframe(Config.DataDir + "/test.txt", "test").sample(
        frac=1, random_state=73
    )
    df = pd.concat([train_map, test_map], axis=0).sample(frac=1, random_state=73)

    # Display sample images
    print_images(df[df["Label"] == "NORMAL"].iloc[0:4], save_path="/report/")
    print_images(df[df["Label"] == "COVID"].iloc[0:4], save_path="/report/")

    # Split the dataframe into train and test/validation
    train_df, test_val_df = train_test_split(
        df, test_size=0.3, random_state=seed, stratify=df["Label"]
    )

    # Split the test/validation dataframe into test and validation
    test_df, val_df = train_test_split(
        test_val_df, test_size=0.5, random_state=seed, stratify=test_val_df["Label"]
    )

    classes = ("COVID", "NORMAL")
    CATEGORIES = sorted(classes)

    # Create data generators for training, validation, and testing
    train_df = train_map.sample(frac=1, random_state=seed)
    train_generator = create_gen(train_df, batch_size=32, seed=42, IMG_SIZE=(224, 224))
    val_generator = create_gen(val_df, batch_size=32, seed=42, IMG_SIZE=(224, 224))
    test_generator = create_gen(test_df, batch_size=32, seed=42, IMG_SIZE=(224, 224))
    kernel_regularizer = regularizers.l2(0.0001)

    # Build the model
    callbacks = getCallbacks("SwinTransformer")
    s_compiled_model = BuildModel("SwinTransformer")
    s_model, s_history = FitModel(
        model=s_compiled_model,
        name="SwinTransformer",
        callbacks=callbacks,
        train_generator=train_generator,
        val_generator=val_generator,
    )

    y_true = test_generator.labels
    y_pred = predict_label(s_model, test_generator)

    # Plot training history
    plot_training_history(s_history, save_path=Config.training_history_path)

    # Evaluate the model
    evaluate_model(s_model, test_generator, y_true, path=Config.evaluation_path)

    # Display confusion matrix
    display_confusion_matrix(
        y_true, y_pred, save_path=Config.confusion_matrix_save_path
    )

    # Save classification report
    save_classification_report(
        y_true, y_pred, save_path=Config.classification_report_path
    )

    # Create the model directory if it doesn't exist
    os.makedirs(Config.model_dir, exist_ok=True)

    # Save the best model
    save_path = os.path.join(Config.model_dir, "best_model.h5")
    s_model.save(save_path)
    print(f"Best model saved at: {save_path}")


if __name__ == "__main__":
    main()
