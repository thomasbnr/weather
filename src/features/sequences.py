import numpy as np

def create_sequences(scaled_data, window=7):
    """
    Create time sequences for LSTM training.
    Input: full feature matrix
    Target: mean temperature (column index 2)
    """

    X, y = [], []

    for i in range(len(scaled_data) - window):
        X.append(scaled_data[i:i+window])
        y.append(scaled_data[i+window][2])  # mean temperature

    X = np.array(X)
    y = np.array(y)

    split_index = int(len(X) * 0.8)

    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test
