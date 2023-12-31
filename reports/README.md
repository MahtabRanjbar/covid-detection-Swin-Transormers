
# Evaluation Metrics
The evaluation metrics for the covid-19 detection model are as follows:

Accuracy: The accuracy metric measures the overall correctness of the model's predictions by calculating the ratio of correctly classified samples to the total number of samples. The accuracy achieved by the model is 0.97 .

F1 Score: The F1 score is the harmonic mean of precision and recall. It provides a balanced measure of the model's performance by considering both false positives and false negatives. The F1 score achieved by the model is 0.97.

Recall: Also known as the true positive rate or sensitivity, recall measures the proportion of actual positive samples that are correctly identified by the model. The recall achieved by the model is 0.97.

Precision: Precision is the ratio of true positives to the sum of true positives and false positives. It indicates the model's ability to correctly classify positive samples. The precision achieved by the model is 0.97.


|          | precision | recall | f1-score | support |
| -------- | --------- | ------ | -------- | ------- |
|    0    |   0.97   |  0.97  |   0.97   |   2429   |
|   1    |   0.97   |  0.96  |   0.97   |   2129    |
| accuracy |           |        |   0.97   |   4558    |
| macro avg|   0.97   |  0.97  |   0.97  |   4558    |
|weighted avg|  0.97    |  0.97  |   0.97   |   4558   |

-----

## Confusion Matrix

The confusion matrix provides a visual representation of the model's performance by showing the count of true positive, true negative, false positive, and false negative predictions.

![Confusion Matrix](/reports/confusion_matrix.png)

## Training History 
The training history plot shows the model's accuracy and loss values over the epochs.

![Accuracy and Loss Plot](/reports/training_history.png)