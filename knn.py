import numpy as np
import matplotlib.pyplot as plt

def distance_matrix(X, Y):
    X_square = np.sum(X**2, axis=1).reshape(-1, 1)
    Y_square = np.sum(Y**2, axis=1).reshape(1, -1)
    XY = np.dot(X, Y.T)
    dists = np.sqrt(X_square + Y_square - 2 * XY)
    return dists

def knn_predict(dists, labels_train, k):
    n_test = dists.shape[0]
    labels_pred = np.zeros(n_test, dtype=labels_train.dtype)
    
    for i in range(n_test):
        # Find the indices of the k nearest neighbors
        nearest_neighbors = np.argsort(dists[i])[:k]
        # Get the labels of the k nearest neighbors
        nearest_labels = labels_train[nearest_neighbors]
        # Predict the label by majority vote
        labels_pred[i] = np.bincount(nearest_labels).argmax()
    
    return labels_pred

def evaluate_knn(data_train, labels_train, data_test, labels_test, k):
    # Compute the distance matrix between the test and training data
    dists = distance_matrix(data_test, data_train)
    
    # Predict the labels for the test set
    labels_pred = knn_predict(dists, labels_train, k)
    
    # Compute the accuracy
    accuracy = np.mean(labels_pred == labels_test)
    
    return accuracy