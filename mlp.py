import numpy as np
from tqdm import tqdm
from scipy.special import expit, log1p

def learn_once_mse(w1, b1, w2, b2, data, targets, learning_rate):

    # Forward pass
    a0 = data # the data are the input of the first layer
    z1 = np.dot(a0, w1) + b1  # input of the hidden layer
    a1 = 1 / (1 + np.exp(-z1))  # output of the hidden layer (sigmoid activation function)
    z2 = np.dot(a1, w2) + b2  # input of the output layer
    a2 = 1 / (1 + np.exp(-z2))  # output of the output layer (sigmoid activation function)
    predictions = a2  # the predicted values are the outputs of the output layer

    # Compute loss (MSE)
    loss = np.mean(np.square(predictions - targets))
    
    # Backward pass
    dloss_dpredictions = 2 * (predictions - targets) / len(data)

    dpredictions_dz2 = a2 * (1 - a2)

    dloss_dz2 = dloss_dpredictions * dpredictions_dz2

    dz2_dw2 = a1
    dloss_dw2 = np.dot(dz2_dw2.T, dloss_dz2)
    dloss_db2 = np.sum(dloss_dz2, axis=0)

    dz2_da1 = w2
    dloss_da1 = np.dot(dloss_dz2, dz2_da1.T)

    da1_dz1 = a1 * (1 - a1)
    dloss_dz1 = dloss_da1 * da1_dz1

    dz1_dw1 = a0
    dloss_dw1 = np.dot(dz1_dw1.T, dloss_dz1)
    dloss_db1 = np.sum(dloss_dz1, axis=0)

    # Update weights and biases
    w2 -= learning_rate * dloss_dw2
    b2 -= learning_rate * dloss_db2
    w1 -= learning_rate * dloss_dw1
    b1 -= learning_rate * dloss_db1

    return w1, b1, w2, b2, loss



def one_hot(array):
    num_classes = np.max(array) + 1
    one_hot_matrix = np.eye(num_classes)[array]
    return one_hot_matrix


def learn_once_cross_entropy(w1, b1, w2, b2, data, labels_train, learning_rate):
    # Forward pass
    a0 = data  # input for the first layer
    z1 = np.dot(a0, w1) + b1  # input of the hidden layer
    a1 = expit(z1)  # use of scipy sigmoid funciton to better handle overflow situation
    z2 = np.dot(a1, w2) + b2  # input of the output layer
    # use of log-sum-exp trick to compute softmax without overflow
    z2_max = np.max(z2, axis=1, keepdims=True)
    a2 = np.exp(z2 - z2_max) / np.sum(np.exp(z2 - z2_max), axis=1, keepdims=True)
    predictions = a2

    # One-hot encode the labels
    targets = one_hot(labels_train)

    # Compute loss (binary cross-entropy) and control predictions value to avoid overflow in log
    epsilon = 1e-15  # Small constant to avoid log(0)
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    loss = -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    accuracy = np.mean(np.argmax(predictions, axis=1) == labels_train)

    # Backward pass
    dloss_dz2 = predictions - targets

    dz2_dw2 = a1
    dloss_dw2 = np.dot(dz2_dw2.T, dloss_dz2)
    dloss_db2 = np.sum(dloss_dz2, axis=0)

    dz2_da1 = w2
    dloss_da1 = np.dot(dloss_dz2, dz2_da1.T)

    da1_dz1 = a1 * (1 - a1)
    dloss_dz1 = dloss_da1 * da1_dz1

    dz1_dw1 = a0
    dloss_dw1 = np.dot(dz1_dw1.T, dloss_dz1)
    dloss_db1 = np.sum(dloss_dz1, axis=0)

    # Update weights and biases
    w2 -= learning_rate * dloss_dw2
    b2 -= learning_rate * dloss_db2
    w1 -= learning_rate * dloss_dw1
    b1 -= learning_rate * dloss_db1

    return w1, b1, w2, b2, loss, accuracy


def train_mlp(w1, b1, w2, b2, data_train, labels_train, learning_rate, num_epoch):
    train_accuracies = []

    for epoch in tqdm(range(num_epoch), desc="Training Progress"):
        w1, b1, w2, b2, loss, accuracy = learn_once_cross_entropy(w1, b1, w2, b2, data_train, labels_train, learning_rate)
        
        train_accuracies.append(accuracy)

    return w1, b1, w2, b2, train_accuracies

def test_mlp(w1, b1, w2, b2, data_test, labels_test):
    # Forward pass
    a0 = data_test
    z1 = np.dot(a0, w1) + b1
    a1 = expit(z1)

    z2 = np.dot(a1, w2) + b2
    #a2 = expit(z2)
    z2_max = np.max(z2, axis=1, keepdims=True)
    a2 = np.exp(z2 - z2_max) / np.sum(np.exp(z2 - z2_max), axis=1, keepdims=True)

    predictions = np.argmax(a2, axis=1)
    test_accuracy = np.mean(predictions == labels_test)
    
    return test_accuracy

def run_mlp_training(data_train, labels_train, data_test, labels_test, d_h, learning_rate, num_epoch):
    # Initialize weights and biases
    input_dim = data_train.shape[1]
    output_dim = np.max(labels_train) + 1
    
    w1 = np.random.randn(input_dim, d_h)
    b1 = np.zeros(d_h)
    w2 = np.random.randn(d_h, output_dim)
    b2 = np.zeros(output_dim)
    
    # Train the MLP
    print("Training the MLP...")
    w1, b1, w2, b2, train_accuracies = train_mlp(w1, b1, w2, b2, data_train, labels_train, learning_rate, num_epoch)
    
    # Test the MLP
    test_accuracy = test_mlp(w1, b1, w2, b2, data_test, labels_test)
    
    return train_accuracies, test_accuracy

