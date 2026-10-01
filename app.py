import os
import time
import joblib
import requests
import pandas as pd
import gradio as gr


# =========================
# Load Model
# =========================
model = joblib.load("frost_random_forest.pkl")
model_info = joblib.load("frost_model_info.pkl")

features = model_info["features"]
threshold = model_info.get("threshold", 0.20)


# =========================
# Weather Cache
# =========================
weather_cache = {}

CACHE_DURATION = 15 * 60  # 15 minutes


# =========================
# Get Weather
# =========================
def get_weather(latitude, longitude):

    cache_key = (round(latitude, 4), round(longitude, 4))
    current_time = time.time()

    # Use cached data if available
    if cache_key in weather_cache:
        cached_time, cached_data = weather_cache[cache_key]

        if current_time - cached_time < CACHE_DURATION:
            return cached_data.copy()

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "precipitation",
            "cloud_cover",
            "wind_speed_10m"
        ],
        "timezone": "Asia/Amman",
        "forecast_days": 3
    }

    # Retry if API temporarily returns 429
    for attempt in range(3):

        try:
            response = requests.get(
                url,
                params=params,
                timeout=30
            )

            if response.status_code == 429:

                if attempt < 2:
                    wait_time = 10 * (attempt + 1)
                    time.sleep(wait_time)
                    continue

                raise Exception(
                    "خدمة الطقس مشغولة حاليًا. حاول مرة أخرى بعد دقائق."
                )

            response.raise_for_status()

            data = response.json()

            weather = pd.DataFrame(data["hourly"])

            # Save to cache
            weather_cache[cache_key] = (
                current_time,
                weather.copy()
            )

            return weather

        except requests.exceptions.RequestException as e:

            if attempt < 2:
                time.sleep(5)
                continue

            raise Exception(
                "تعذر الاتصال بخدمة الطقس حاليًا. حاول مرة أخرى لاحقًا."
            ) from e

    raise Exception("تعذر الحصول على بيانات الطقس.")
