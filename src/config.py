class Config:
    # Paths
    DataDir = "/kaggle/input/covidx-cxr2"
    evaluation_path = "reports/evaluation_metrics.txt"
    training_history_path = "reports/training_history.png"
    confusion_matrix_save_path = "reports/confusion_matrix.png"
    classification_report_path = "reports/classification_report.txt"
    model_dir = "saved_models"

    # Model training configuration
    seed = 42
    IMAGE_SIZE = (224, 224)
    final_activation = "softmax"
    entropy = "sparse_categorical_crossentropy"
    n_classes = 2
    EPOCHS = 10  # 120
    patience = 3
    start_lr = 0.00001
    min_lr = 0.00001
    max_lr = 0.00005
    rampup_epochs = 5
    sustain_epochs = 0
    exp_decay = 0.8
    ColorCh = 3
    IMG_SIZE = 224
    input_shape = (IMG_SIZE, IMG_SIZE, ColorCh)