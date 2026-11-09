# MNIST Digit Classification with Deep Neural Networks

## 1. Workflow
The project follows a structured machine learning pipeline designed to construct, evaluate, and optimize a Deep Neural Network (DNN) for digit classification using the MNIST dataset:
*   **Environment Setup:** Initialized necessary libraries (TensorFlow, NumPy, Matplotlib, Pandas, Seaborn, scikit-learn) and established global random seeds to ensure experiment reproducibility.
*   **Data Preparation:** Loaded the MNIST dataset, reshaped the 28x28 pixel images into 1D arrays (784 features), and applied one-hot encoding to the target labels.
*   **Exploratory Data Analysis:** Visualized random samples from the training set to verify data integrity and class representation.
*   **Baseline Modeling:** Constructed an initial Sequential model to establish baseline performance metrics.
*   **Hyperparameter Tuning:** Developed a secondary "playground" model to experiment with neuron counts, learning rates, epochs, and activation functions.
*   **Validation & Error Analysis:** Evaluated the tuned model against unseen test data. Generated a confusion matrix and plotted specific misclassified digits alongside their true labels to identify the model's weak points.
*   **Workflow Improvements:** Implemented a final optimized architecture integrating data normalization (rescaling), Batch Normalization layers to stabilize training, and an Early Stopping callback to prevent overfitting and optimize computational cost.

## 2. Final Model Architecture
The optimized neural network utilizes a Sequential architecture designed for robust feature extraction and classification:
*   **Input:** 784-dimensional vector (flattened 28x28 images).
*   **Preprocessing Layer:** Rescaling Layer (scales pixel values by `1./255`).
*   **Hidden Layer 1:** Dense Layer (512 units, ReLU activation).
*   **Normalization 1:** Batch Normalization Layer.
*   **Hidden Layer 2:** Dense Layer (512 units, ReLU activation).
*   **Normalization 2:** Batch Normalization Layer.
*   **Output Layer:** Dense Layer (10 units, Softmax activation).

## 3. Selected Hyperparameters
The final model configuration leverages the following hyperparameter suite, determined through iterative tuning and performance monitoring:
*   **Optimizer:** Stochastic Gradient Descent (SGD)
*   **Learning Rate:** 0.01
*   **Loss Function:** Categorical Cross-Entropy
*   **Metrics Monitored:** Accuracy
*   **Maximum Epochs:** 50
*   **Early Stopping:** Monitored on `val_loss` with a patience of 5 epochs (restoring best weights upon completion).
*   **Hidden Activation:** ReLU (Rectified Linear Unit)
*   **Output Activation:** Softmax

## 4. Requirements
The following specifications and library versions reflect the development environment used for training and deployment:

```text
# Python 3.10 environment

tensorflow==2.15.0
numpy
pandas
matplotlib
seaborn
scikit-learn
