import numpy as np

def create_sequences(scaled_data, window_hours=168):
    """
    Create sequences of 168 hours (7 days) for next-hour prediction.
    window_hours: number of hours per sequence.
    """

    X, y = [], []

    for i in range(len(scaled_data) - window_hours):
        X.append(scaled_data[i:i + window_hours])
        y.append(scaled_data[i + window_hours][0])  # column 0 = temperature

    X = np.array(X)
    y = np.array(y)

    split = int(len(X) * 0.8)

    return X[:split], X[split:], y[:split], y[split:]
