import requests
import pandas as pd

def load_weather_data():
    """Download hourly weather data from Open-Meteo for New York."""

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": 40.7128,
        "longitude": -74.0060,
        "start_date": "2010-01-01",
        "end_date": "2024-12-31",
        "hourly": [
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
        ],
        "timezone": "America/New_York"
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "hourly" not in data:
        raise ValueError(f"API response missing 'hourly'. Raw response: {data}")

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    return df
