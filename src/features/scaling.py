from sklearn.preprocessing import MinMaxScaler

def scale_data(df):
    """Scale hourly weather features using MinMaxScaler."""

    feature_columns = [
        "temperature_2m",
        "relativehumidity_2m",
        "dewpoint_2m",
        "windspeed_10m",
        "winddirection_10m",
        "pressure_msl",
        "shortwave_radiation",
        "direct_radiation",
        "diffuse_radiation",
        "cloudcover",
        "precipitation"
    ]

    values = df[feature_columns].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    return scaled, scaler, feature_columns
