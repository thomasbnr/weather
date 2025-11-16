import requests
import pandas as pd

def load_weather_data():
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": 40.7128,
        "longitude": -74.0060,
        "start_date": "2010-01-01",
        "end_date": "2024-12-31",
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "windspeed_10m_max",
            "winddirection_10m_dominant",
            "shortwave_radiation_sum"
        ],
        "timezone": "America/New_York"
    }

    response = requests.get(url, params=params)

    print("STATUS CODE:", response.status_code)
    print("RAW RESPONSE:", response.text[:1000])   # important
    print("JSON KEYS:", list(response.json().keys()))

    if "daily" not in response.json():
        raise ValueError("The API response does NOT contain 'daily'. See above logs.")

    data = response.json()["daily"]

    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    return df
