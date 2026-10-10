
import time
import requests
import pandas as pd

# Cache for successful weather responses
weather_cache = {}

CACHE_DURATION = 30 * 60       # 30 minutes
MAX_STALE_DURATION = 6 * 60 * 60  # 6 hours


def get_weather(latitude, longitude):
    cache_key = (round(latitude, 4), round(longitude, 4))
    now = time.time()

    cached = weather_cache.get(cache_key)

    # 1. Return fresh cached data without calling the API
    if cached:
        cached_time, cached_data = cached
        if now - cached_time < CACHE_DURATION:
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

    last_error = None

    # 2. Retry a limited number of times with backoff
    for attempt in range(3):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=20
            )

            if response.status_code == 429:
                last_error = "Open-Meteo returned HTTP 429"

                if attempt < 2:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        wait = float(retry_after)
                    except (TypeError, ValueError):
                        wait = 5 * (2 ** attempt)

                    # Avoid excessively long blocking waits
                    time.sleep(min(wait, 20))
                    continue

                break

            response.raise_for_status()
            payload = response.json()

            if "hourly" not in payload:
                raise ValueError("Weather response has no hourly data")

            weather = pd.DataFrame(payload["hourly"])

            required_columns = [
                "time",
                "temperature_2m",
                "relative_humidity_2m",
                "dew_point_2m",
                "precipitation",
                "cloud_cover",
                "wind_speed_10m"
            ]

            missing = [
                col for col in required_columns
                if col not in weather.columns
            ]

            if missing:
                raise ValueError(
                    f"Missing weather fields: {missing}"
                )

            # Save only successful responses
            weather_cache[cache_key] = (
                time.time(),
                weather.copy()
            )

            return weather

        except requests.exceptions.RequestException as exc:
            last_error = str(exc)

            if attempt < 2:
                time.sleep(2 * (2 ** attempt))

        except (ValueError, KeyError) as exc:
            last_error = str(exc)
            break

    # 3. If the API fails, use previously successful data
    if cached:
        cached_time, cached_data = cached

        if time.time() - cached_time <= MAX_STALE_DURATION:
            return cached_data.copy()

    raise RuntimeError(
        "تعذر جلب بيانات الطقس حاليًا، ولا توجد بيانات محفوظة "
        "حديثة بما يكفي. انتظري قليلًا ثم حاولي مرة أخرى. "
        f"التفاصيل: {last_error}"
    )
