from data.loader import load_weather_data
from features.scaling import scale_data
from features.sequences import create_sequences
from models.training import train_lstm_model
from models.prediction import predict_next_hour
from utils.paths import MODELS_DIR

import joblib

def main():
    print("Loading hourly weather data...")
    df = load_weather_data()

    print("Scaling data...")
    scaled, scaler, features = scale_data(df)

    print("Creating sequences...")
    X_train, X_test, y_train, y_test = create_sequences(scaled, window_hours=168)

    print("Training model...")
    model = train_lstm_model(X_train, y_train, X_test, y_test)

    print("Saving model...")
    model.save(f"{MODELS_DIR}/hourly_lstm_model.h5")
    joblib.dump(scaler, f"{MODELS_DIR}/hourly_scaler.pkl")

    print("Predicting next hour temperature...")
    pred = predict_next_hour(model, scaler, scaled)
    print(f"Predicted next-hour temperature: {pred:.2f} °C")

if __name__ == "__main__":
    main()
