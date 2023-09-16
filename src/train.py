import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (Callback, EarlyStopping,
                                        LearningRateScheduler, ModelCheckpoint,
                                        ReduceLROnPlateau, TensorBoard)

from config import Config
from model import SwinTransformer


def getMetrics(type_):
    if type_ == "accuracy":
        return "accuracy"
    if type_ == "loss":
        return "loss"
    if type_ == "val_accuracy":
        return "val_accuracy"
    if type_ == "val_loss":
        return "val_loss"


def load_swin_transformer():
    img_adjust_layer = tf.keras.layers.Lambda(
        lambda data: tf.keras.applications.imagenet_utils.preprocess_input(
            tf.cast(data, tf.float32), mode="torch"
        ),
        input_shape=[*Config.IMAGE_SIZE, 3],
    )
    pretrained_model = SwinTransformer(
        "swin_tiny_224",
        num_classes=2,
        include_top=False,
        pretrained=True,
        use_tpu=True,
    )
    model = tf.keras.Sequential(
        [
            img_adjust_layer,
            pretrained_model,
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )
    return model


def lrfn(epoch):
    """
    Calculate the learning rate for a given epoch based on specified ramp-up, sustain, and decay parameters.

    Args:
        epoch (int): The current epoch for which the learning rate needs to be calculated.

    Returns:
        learning_rate (float): The learning rate for the given epoch.
    """
    if epoch < Config.rampup_epochs:
        return (
            Config.max_lr - Config.start_lr
        ) / Config.rampup_epochs * epoch + Config.start_lr
    elif epoch < Config.rampup_epochs + Config.sustain_epochs:
        return Config.max_lr
    else:
        return (Config.max_lr - Config.min_lr) * Config.exp_decay ** (
            epoch - Config.rampup_epochs - Config.sustain_epochs
        ) + Config.min_lr


def getCallbacks(name):
    """
    Return a list of callback objects commonly used in training neural networks.

    Args:
        name (str): Name or identifier for the callbacks.

    Returns:
        callbacks (list): A list of callback objects to be used during model training.
    """

    class myCallback(Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get(getMetrics("accuracy")) >= 0.999:
                print("\nLimits Reached, cancelling training!")
                self.model.stop_training = True

    end_callback = myCallback()
    lr_plat = ReduceLROnPlateau(patience=2, mode="min")
    lr_callback = LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=False)
    early_stopping = EarlyStopping(
        patience=Config.patience,
        monitor=getMetrics("val_loss"),
        mode="min",
        restore_best_weights=True,
        verbose=1,
        min_delta=0.00075,
    )
    checkpoint_filepath = name + "_Weights.h5"
    model_checkpoints = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor=getMetrics("val_loss"),
        mode="min",
        verbose=1,
        save_best_only=True,
    )
    log_dir = "logs/fit/" + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(
        log_dir=log_dir, write_graph=True, histogram_freq=1
    )

    return [end_callback, lr_callback, model_checkpoints, early_stopping, Config.lr_plat]


callbacks = getCallbacks("SwinTransformer")


def CompileModel(name, model):
    model.compile(optimizer="adam", loss=Config.entropy, metrics=["accuracy"])
    return model


def FitModel(model, name, callbacks, train_generator, val_generator):
    callbacks_ = callbacks

    history = model.fit(
        train_generator,
        epochs=Config.EPOCHS,
        callbacks=callbacks_,
        validation_data=val_generator,
        steps_per_epoch=(len(train_generator.labels) / 80),
        validation_steps=(len(val_generator.labels) / 80),
    )

    model.load_weights(name + "_Weights.h5")

    final_accuracy_avg = np.mean(history.history[getMetrics("val_accuracy")][-5:])

    final_loss = history.history[getMetrics("val_loss")][-1]

    group = {
        history: "history",
        name: "name",
        model: "model",
        final_accuracy_avg: "acc",
        final_loss: "loss",
    }

    print("\n")
    print("---" * 15)
    print(name, " Model")
    print("Total Epochs :", len(history.history[getMetrics("loss")]))
    print("Restoring best Weights")

    index = len(history.history[getMetrics("loss")]) - (Config.patience + 1)
    print("---" * 15)
    print("Best Epoch :", index)
    print("---" * 15)

    train_accuracy = history.history[getMetrics("accuracy")][index]
    train_loss = history.history[getMetrics("loss")][index]

    val_accuracy = history.history[getMetrics("val_accuracy")][index]
    val_loss = history.history[getMetrics("val_loss")][index]

    print("Accuracy on train:", train_accuracy, "\tLoss on train:", train_loss)

    print("Accuracy on val:", val_accuracy, "\tLoss on val:", val_loss)
    print("---" * 15)

    return model, history


def BuildModel(name):
    prepared_model = load_swin_transformer()

    compiled_model = CompileModel(name, prepared_model)
    return compiled_model
