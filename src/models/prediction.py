import numpy as np

def predict_tomorrow(model, scaler, scaled_data):
    """Predict tomorrow's mean temperature using all features."""

    # Extract last 7 days of features
    last_week = scaled_data[-7:]
    last_week = np.expand_dims(last_week, axis=0)

    # Predict normalized value
    scaled_prediction = model.predict(last_week)[0][0]

    # Build zero vector to inverse transform only the target feature
    dummy = np.zeros((1, scaled_data.shape[1]))
    dummy[0][2] = scaled_prediction  # column 2 = temperature mean

    # Convert back to real temperature
    prediction = scaler.inverse_transform(dummy)[0][2]

    return prediction
