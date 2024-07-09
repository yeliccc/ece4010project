import numpy as np
from sklearn.metrics import mean_squared_error


def calculate_rmse(predictions, actual):
    predictions = predictions[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return np.sqrt(mean_squared_error(predictions, actual))


def predict_ratings(matrix, similarity, type='user'):
    if type == 'user':
        return similarity.dot(matrix) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        return matrix.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
