# CIFAR-10 Image Classification

[![Python](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview
This project implements and compares two classic machine learning models for image classification on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset:
- **K-Nearest Neighbors (KNN)**
- **Multilayer Perceptron (MLP)**

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The goal is to evaluate and compare the performance of a simple, non-parametric method (KNN) and a basic neural network (MLP) on this challenging image classification task.

---

## Project Structure
- [`read_cifar.py`](read_cifar.py): Functions to load and preprocess the CIFAR-10 dataset
- [`knn.py`](knn.py): Implementation of the K-Nearest Neighbors algorithm
- [`mlp.py`](mlp.py): Implementation of the Multilayer Perceptron neural network
- [`test.ipynb`](test.ipynb): Jupyter notebook for training, testing, and visualizing the models
- [`results/`](results/): Directory containing performance plots and results

---

## Setup & Installation
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd image-classification
   ```
2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install numpy matplotlib
   ```
3. **Download CIFAR-10 dataset:**
   - Download the CIFAR-10 Python version from [here](https://www.cs.toronto.edu/~kriz/cifar.html).
   - Extract the files into a `data/` directory at the project root.

---

## Usage
To reproduce the results and visualize model performance:

1. **Ensure the CIFAR-10 dataset is in the `data/` directory.**
2. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook test.ipynb
   ```
3. **Run all cells** in the notebook to train and evaluate both models.
4. **View generated plots** in the `results/` directory after execution.

---

## Results
The project compares the performance of both models:

### KNN Model
- Tested with k values from 1 to 20
- **Best accuracy:** ~33.5% (at k=5, excluding k=1)
- ![KNN Results](results/knn.png)

### MLP Model
- Hidden layer size: 1024 neurons
- Trained for 100 epochs, learning rate: 0.01
- **Test accuracy:** ~21%
- ![MLP Results](results/mlp.png)

**Conclusion:**
> The KNN implementation outperformed the MLP model for this dataset and configuration, achieving an accuracy of 33.5% compared to the MLP's 21%.

---

## Requirements
- Python 3.x
- NumPy
- Matplotlib
- Pickle (standard library, for reading CIFAR-10 data)

---

*Developed by Stevan Le Stanc*