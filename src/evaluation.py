# import data handling tools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# import Deep learning Libraries
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def predict_label(model, test_generator):
    predictions = model.predict(test_generator, verbose=1)
    preds = np.argmax(predictions, axis=1)
    return preds


def evaluate_model(model, test_images, y_test, path=None):
    """
    Evaluates the model by predicting labels for test images and calculating various metrics.

    Args:
        model: The trained model to evaluate.
        test_images: The test images to make predictions on.
        y_test: The true labels of the test_images.
        path (optional): The path to save the evaluation metrics. If provided, the metrics will be saved to the specified location.

    Returns:
        A dictionary containing the evaluation results, including predictions, accuracy, F1 score, recall, and precision.
    """
    # Predict the labels of the test_images
    pred = model.predict(test_images)
    pred = np.argmax(pred, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="weighted")
    recall = recall_score(y_test, pred, average="weighted")
    precision = precision_score(y_test, pred, average="weighted")

    evaluation_results = {
        "predictions": pred,
        "accuracy": accuracy,
        "f1_score": f1,
        "recall": recall,
        "precision": precision,
    }

    # Print the evaluation metrics
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Recall:", recall)
    print("Precision:", precision)

    if path is not None:
        # Write the evaluation metrics to a text file
        with open(path, "w") as file:
            file.write(f"Accuracy: {accuracy}\n")
            file.write(f"F1 Score: {f1}\n")
            file.write(f"Recall: {recall}\n")
            file.write(f"Precision: {precision}\n")

    return evaluation_results


def plot_training_history(history, save_path=None):
    """
    Plots the training history of a model, including training and validation loss, and training and validation accuracy.

    Args:
        history: The training history object returned by the `fit` method of a Keras model.
        save_path (optional): The path to save the plot. If provided, the plot will be saved to the specified location.

    Returns:
        None
    """
    # Define needed variables
    tr_acc = history.history["accuracy"]
    tr_loss = history.history["loss"]
    val_acc = history.history["val_accuracy"]
    val_loss = history.history["val_loss"]
    index_loss = np.argmin(val_loss)
    val_lowest = val_loss[index_loss]
    index_acc = np.argmax(val_acc)
    acc_highest = val_acc[index_acc]
    Epochs = [i + 1 for i in range(len(tr_acc))]
    loss_label = f"best epoch= {str(index_loss + 1)}"
    acc_label = f"best epoch= {str(index_acc + 1)}"

    # Plot training history
    plt.figure(figsize=(20, 8))
    plt.style.use("fivethirtyeight")

    plt.subplot(1, 2, 1)
    plt.plot(Epochs, tr_loss, "r", label="Training loss")
    plt.plot(Epochs, val_loss, "g", label="Validation loss")
    plt.scatter(index_loss + 1, val_lowest, s=150, c="blue", label=loss_label)
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(Epochs, tr_acc, "r", label="Training Accuracy")
    plt.plot(Epochs, val_acc, "g", label="Validation Accuracy")
    plt.scatter(index_acc + 1, acc_highest, s=150, c="blue", label=acc_label)
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()

    # Save the plot if save_path is not None
    if save_path is not None:
        plt.savefig(save_path)


def display_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Display a confusion matrix and save it as an image file if the save_path is not None.

    Args:
        y_true (list): List of true labels.
        y_pred (list): List of predicted labels.
        save_path (str or None): Path to save the confusion matrix image. If None, the image will not be saved.

    Returns:
        None
    """
    CATEGORIES = sorted(("COVID", "NORMAL"))
    cf_matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cf_matrix,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=CATEGORIES,
        yticklabels=CATEGORIES,
    ),

    plt.title("Confusion Matrix\n", fontsize=15)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=6)

    if save_path is not None:
        plt.savefig(save_path)
        print(f"Confusion matrix saved as {save_path}")
    else:
        plt.tight_layout()
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.show()


def save_classification_report(y_true, y_pred, save_path=None):
    """
    Print the classification report and save it as a text file if the save_path is not None.

    Args:
        y_true (list): List of true labels.
        y_pred (list): List of predicted labels.
        save_path (str or None): Path to save the classification report. If None, the report will not be saved.

    Returns:
        None
    """
    report = classification_report(y_true, y_pred)

    print(report)

    if save_path is not None:
        with open(save_path, "w") as file:
            file.write(report)
        print(f"Classification report saved as {save_path}")
