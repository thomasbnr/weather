from sklearn.preprocessing import MinMaxScaler

def scale_data(df):
    """Scale only the daily weather features available in the API response."""

    feature_columns = [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",
        "windspeed_10m_max",
        "winddirection_10m_dominant",
        "shortwave_radiation_sum"
    ]

    values = df[feature_columns].values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(values)

    return scaled_data, scaler
