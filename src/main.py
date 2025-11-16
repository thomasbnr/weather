from data.loader import load_weather_data
from features.scaling import scale_data
from features.sequences import create_sequences
from models.training import train_lstm_model
from models.prediction import predict_tomorrow
from utils.paths import MODELS_DIR

import joblib

def main():
    print("Loading weather data...")
    df = load_weather_data()

    print("Scaling data...")
    scaled_data, scaler = scale_data(df)

    print("Creating sequences...")
    X_train, X_test, y_train, y_test = create_sequences(scaled_data, window=7)

    print("Training LSTM model...")
    model = train_lstm_model(X_train, y_train, X_test, y_test)

    print("Saving model and scaler...")
    model.save(f"{MODELS_DIR}/weather_lstm_model.h5")
    joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")

    print("Predicting tomorrow's temperature...")
    prediction = predict_tomorrow(model, scaler, scaled_data)

    print("Predicted average temperature for tomorrow:", round(prediction, 2), "°C")

if __name__ == "__main__":
    main()
