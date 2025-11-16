import numpy as np

def predict_next_hour(model, scaler, scaled_data):
    """Predict next hour temperature using last 168 hours."""

    last_seq = scaled_data[-168:]
    last_seq = np.expand_dims(last_seq, axis=0)

    scaled_pred = model.predict(last_seq)[0][0]

    dummy = np.zeros((1, scaled_data.shape[1]))
    dummy[0][0] = scaled_pred

    prediction = scaler.inverse_transform(dummy)[0][0]

    return prediction
