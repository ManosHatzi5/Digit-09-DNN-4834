

---

# 🧠 MNIST Digit Classification Neural Network

A multi-class deep learning model built to accurately classify handwritten digits from the MNIST dataset. This project focuses on implementing robust training controls and rigorous model evaluation techniques.

## 📖 Overview

This repository contains a complete pipeline for loading image data, defining a dense neural network architecture, training the model with specific optimization constraints, and visualizing the final predictive performance.

## ✨ Key Features

*   **Deep Neural Network Architecture:** Built using TensorFlow and Keras to process $28 \times 28$ pixel grayscale images into 10 distinct digit classes.
*   **Early Stopping Implementation:** Integrates Keras callbacks to monitor validation loss during training, automatically halting epochs when performance degrades to prevent overfitting.
*   **Performance Visualization:** Generates a detailed confusion matrix upon test evaluation, allowing for granular analysis of false positives and classification bottlenecks across specific digits.
*   **Optimized Pipeline:** Handles data normalization, one-hot encoding, and batch processing efficiently.

## 🛠️ Tech Stack

*   **Language:** Python
*   **Machine Learning Framework:** TensorFlow, Keras
*   **Data Processing & Visualization:** NumPy, Matplotlib, Scikit-learn

## 🏁 Getting Started

1. Clone the repository:
   ```bash
   git clone <your-repo-link>
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
