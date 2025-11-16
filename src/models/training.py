import matplotlib.pyplot as plt
from models.lstm_model import build_lstm_model

def train_lstm_model(X_train, y_train, X_test, y_test):
    """Train the LSTM model and display the loss curve."""

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=32
    )

    plt.plot(history.history["loss"], label="training")
    plt.plot(history.history["val_loss"], label="validation")
    plt.legend()
    plt.title("Loss Curve")
    plt.show()

    return model
