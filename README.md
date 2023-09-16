# COVID-19 Detection  using Swin Transformers
This project aims to detect COVID-19 cases from chest X-ray images using the Swin Transformer model. The model is trained on a dataset of chest X-ray images obtained from the Chest X-ray Dataset on Kaggle. By leveraging the power of Swin Transformers, the model achieves accurate classification of X-ray images into two categories: COVID-19 positive or normal.



---- 
## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model](#model)
- [Evaluation Metrics](#evaluation-metrics)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
---

## Installation
1. Clone this repository to your local machine:

   ```sh
   git clone https://github.com/Rozajalilipour/TumorDetectAI.git
   ```

2. Navigate to the project directory:

   ```sh
   cd Brain-tumor-detection-app

   ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```
---
## Usage

Usage
To run the project, follow these steps:

1. Set the necessary configurations in the config.py file. Update the DataDir variable to the directory where the dataset is located.

2. Run the main script:
    ```sh
    python src/main.py
    ```
3. The script will load and preprocess the image data, split it into training and testing sets, and create data generators for training, validation, and testing.

4. The Swin Transformer model will be built and trained using the training data. The model will be evaluated using the testing data, and the performance metrics will be displayed, including a confusion matrix and classification report.

5. The best-performing model will be saved in the model_dir directory as best_model.h5.

6. for ease of use and not installing all neccessary packages and having no conflict, you can see the  [notebook](notebooks/braintumordetection.ipynb) of project
---
## Dataset
The dataset used for training and testing the model is the Chest X-ray Dataset from [Kaggle](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection). It contains a collection of chest X-ray images in two categories: COVID-19 positive and normal. To use this dataset, download the dataset from provided link.

The dataset has been split into three subsets: train set, validation set, and test set. The sizes of each subset are as follows:

- Train set size: 161
- Validation set size: 41
- Test set size: 51
---
## Model

The Swin Transformer model is a state-of-the-art architecture that combines the strengths of transformers and convolutional neural networks (CNNs) for image classification tasks. It introduces a hierarchical architecture that divides the image into patches and processes them with self-attention mechanisms. This allows the model to capture global and local dependencies effectively, enabling robust feature representation.

The Swin Transformer model used in this project is implemented using TensorFlow and Keras. It consists of several layers, including patch embedding, multi-scale self-attention, and feed-forward networks.

---
## Evaluation Metrics
After training and testing the brain tumor detection model, several evaluation metrics have been computed to assess its performance. These evaluation metrics provide insights into the model's effectiveness in detecting covid-19. The metrics include accuracy, F1 score, recall, and precision.


| Metric     | Value     |
|------------|-----------|
| Accuracy   | 0.9019    |
| F1 Score   | 0.9002    |
| Recall     | 0.9019    |
| Precision  | 0.9054    |

For further evaluation metrics and details, please check the [reports](reports/README.md)  folder.



## Contributing
Contributions are always welcome! If you have any ideas or suggestions, please feel free to open an issue or a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information


## Contact
If you have any questions or comments about this project, please feel free to contact me at mahtabranjbar93@gmail.com








