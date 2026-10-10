
# -*- coding: utf-8 -*-
import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone

import gradio as gr
import joblib
import pandas as pd
import requests


# ==========================================================
# 1. Model and Configuration
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "frost_random_forest.pkl"
INFO_PATH = BASE_DIR / "frost_model_info.pkl"
CACHE_FILE = BASE_DIR / "weather_cache.json"

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
MET_NORWAY_URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact"

CACHE_DURATION = 6 * 60 * 60
MAX_STALE_DURATION = 48 * 60 * 60
REQUEST_TIMEOUT = 15

if not MODEL_PATH.exists() or not INFO_PATH.exists():
    raise FileNotFoundError(
        "تأكدي من وجود frost_random_forest.pkl "
        "و frost_model_info.pkl بجانب app.py."
    )

model = joblib.load(MODEL_PATH)
model_info = joblib.load(INFO_PATH)

features = list(model_info["features"])
print("[MODEL FEATURES]", features, flush=True)
threshold = float(model_info.get("threshold", 0.20))

# Shared HTTP session.
session = requests.Session()
session.headers.update({
    "User-Agent": "JordanFrostWarning/1.0 (weather forecast application)"
})


# ==========================================================
# 2. Persistent Local Cache
# ==========================================================
def load_cache():
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)
                return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[CACHE] Load error: {exc}", flush=True)

    return {}


weather_cache = load_cache()


def save_cache():
    temp_file = CACHE_FILE.with_suffix(".tmp")

    try:
        with open(temp_file, "w", encoding="utf-8") as file:
            json.dump(weather_cache, file, ensure_ascii=False)

        os.replace(temp_file, CACHE_FILE)

    except (OSError, TypeError, ValueError) as exc:
        print(f"[CACHE] Save error: {exc}", flush=True)


def cache_key_for(latitude, longitude):
    return f"{float(latitude):.4f},{float(longitude):.4f}"


def get_cached_weather(key):
    entry = weather_cache.get(key)

    if not isinstance(entry, dict):
        return None, None

    try:
        timestamp = float(entry["timestamp"])
        weather = pd.DataFrame(entry["data"])

        if weather.empty:
            return None, None

        return timestamp, weather

    except (KeyError, TypeError, ValueError):
        return None, None


def store_weather(key, weather):
    weather_cache[key] = {
        "timestamp": time.time(),
        "data": weather.to_dict(orient="records"),
    }
    save_cache()


# ==========================================================
# 3. Primary Weather Provider: Open-Meteo
# ==========================================================
def get_open_meteo(latitude, longitude):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "dew_point_2m,"
            "precipitation,"
            "cloud_cover,"
            "wind_speed_10m"
        ),
        "timezone": "Asia/Amman",
        "forecast_days": 3,
    }

    response = session.get(
        OPEN_METEO_URL,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    print(
        f"[WEATHER] Open-Meteo HTTP {response.status_code}",
        flush=True,
    )

    response.raise_for_status()
    payload = response.json()

    if "hourly" not in payload:
        raise ValueError("Open-Meteo did not return hourly data.")

    weather = pd.DataFrame(payload["hourly"])

    required = [
        "time",
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "precipitation",
        "cloud_cover",
        "wind_speed_10m",
    ]

    missing = [column for column in required
               if column not in weather.columns]

    if missing or weather.empty:
        raise ValueError(
            f"Open-Meteo forecast fields missing: {missing}"
        )

    return weather[required].copy()


# ==========================================================
# 4. Alternative Weather Provider: MET Norway
# ==========================================================
def get_met_norway(latitude, longitude):
    params = {
        "lat": latitude,
        "lon": longitude,
    }

    headers = {
        "User-Agent": (
            "JordanFrostWarning/1.0 "
            "(Jordan agricultural frost forecast application)"
        )
    }

    response = session.get(
        MET_NORWAY_URL,
        params=params,
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )

    print(
        f"[WEATHER] MET Norway HTTP {response.status_code}",
        flush=True,
    )

    response.raise_for_status()
    payload = response.json()

    timeseries = (
        payload.get("properties", {}).get("timeseries", [])
    )

    if not timeseries:
        raise ValueError("Alternative provider returned no forecast.")

    rows = []

    for item in timeseries:
        time_text = item.get("time")
        details = (
            item.get("data", {})
            .get("instant", {})
            .get("details", {})
        )

        if not time_text or not details:
            continue

        # The provider returns UTC timestamps.
        try:
            parsed_time = datetime.fromisoformat(
                time_text.replace("Z", "+00:00")
            )
            local_time = parsed_time.astimezone(
                timezone.utc
            ).astimezone(
                __import__("zoneinfo").ZoneInfo("Asia/Amman")
            )
            formatted_time = local_time.strftime("%Y-%m-%dT%H:%M")
        except (ValueError, TypeError):
            formatted_time = time_text

        temperature = details.get("air_temperature")
        humidity = details.get("relative_humidity")
        dew_point = details.get("dew_point_temperature")
        cloud_cover = details.get("cloud_area_fraction")
        wind_speed = details.get("wind_speed")

        next_data = item.get("data", {}).get("next_1_hours", {})
        precipitation = (
            next_data.get("details", {}).get("precipitation_amount")
        )

        # Do not invent missing measurements.
        required_values = [
            temperature,
            humidity,
            dew_point,
            cloud_cover,
            wind_speed,
            precipitation,
        ]

        if any(value is None for value in required_values):
            continue

        rows.append({
            "time": formatted_time,
            "temperature_2m": float(temperature),
            "relative_humidity_2m": float(humidity),
            "dew_point_2m": float(dew_point),
            "precipitation": float(precipitation),
            "cloud_cover": float(cloud_cover),
            "wind_speed_10m": float(wind_speed) * 3.6,
        })

    weather = pd.DataFrame(rows)

    if weather.empty:
        raise ValueError(
            "البيانات البديلة لا تحتوي على جميع القياسات اللازمة للموديل."
        )

    weather = weather.drop_duplicates(subset=["time"])
    weather = weather.sort_values("time").reset_index(drop=True)

    return weather


# ==========================================================
# 5. Weather Retrieval and Fallback
# ==========================================================
def get_weather(latitude, longitude):
    latitude = float(latitude)
    longitude = float(longitude)

    key = cache_key_for(latitude, longitude)
    now = time.time()

    cached_timestamp, cached_weather = get_cached_weather(key)

    # Avoid contacting providers if a fresh forecast is available.
    if (
        cached_weather is not None
        and cached_timestamp is not None
        and 0 <= now - cached_timestamp < CACHE_DURATION
    ):
        print(f"[CACHE] Fresh forecast used: {key}", flush=True)
        return cached_weather.copy()

    errors = []

    # Try the primary provider first.
    try:
        weather = get_open_meteo(latitude, longitude)
        store_weather(key, weather)
        print("[WEATHER] Using Open-Meteo forecast.", flush=True)
        return weather

    except (
        requests.exceptions.RequestException,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        errors.append(f"Open-Meteo: {exc}")
        print(f"[WEATHER] Primary provider failed: {exc}", flush=True)

    # Try the alternative provider only after the primary fails.
    try:
        weather = get_met_norway(latitude, longitude)
        store_weather(key, weather)
        print("[WEATHER] Using MET Norway forecast.", flush=True)
        return weather

    except (
        requests.exceptions.RequestException,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        errors.append(f"MET Norway: {exc}")
        print(f"[WEATHER] Alternative provider failed: {exc}", flush=True)

    # Use a previous successful forecast only within the allowed age.
    cached_timestamp, cached_weather = get_cached_weather(key)

    if cached_weather is not None and cached_timestamp is not None:
        age = time.time() - cached_timestamp

        if 0 <= age <= MAX_STALE_DURATION:
            print(
                f"[CACHE] Using saved forecast; age={age / 3600:.1f}h",
                flush=True,
            )
            return cached_weather.copy()

    raise RuntimeError(
        "تعذر الحصول على بيانات الطقس من المصدرين، "
        "ولا توجد بيانات سابقة صالحة لهذا الموقع. "
        + " | ".join(errors)
    )


# ==========================================================
# 6. Frost Prediction
# ==========================================================
def check_frost_risk(latitude, longitude):
    weather = get_weather(latitude, longitude)

    rename_map = {
        "temperature_2m": "temperature_2m (°C)",
        "relative_humidity_2m": "relative_humidity_2m (%)",
        "dew_point_2m": "dew_point_2m (°C)",
        "precipitation": "precipitation (mm)",
        "cloud_cover": "cloud_cover (%)",
        "wind_speed_10m": "wind_speed_10m (km/h)",
    }

    weather = weather.rename(columns=rename_map).copy()

    missing = [
        feature for feature in features
        if feature not in weather.columns
    ]

    if missing:
        raise ValueError(
            "خصائص الطقس لا تتطابق مع الموديل: "
            f"{missing}"
        )

    X = weather[features].copy()

    # Ensure model inputs are numeric.
    for column in X.columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    valid_rows = X.notna().all(axis=1)

    if not valid_rows.any():
        raise ValueError(
            "لا توجد ساعات تحتوي على قياسات مكتملة للموديل."
        )

    weather = weather.loc[valid_rows].copy()
    X = X.loc[valid_rows]

    probabilities = model.predict_proba(X)
    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            "لم يتم العثور على فئة الصقيع (1) في الموديل."
        )

    frost_index = classes.index(1)
    weather["frost_probability"] = probabilities[:, frost_index]
    weather["frost_risk"] = (
        weather["frost_probability"] >= threshold
    ).astype(int)

    return weather.reset_index(drop=True)


# ==========================================================
# 7. Jordan Locations
# ==========================================================
jordan_locations = {
    "عمّان": {
        "عمّان": (31.9539, 35.9106),
        "ماركا": (31.9833, 35.9833),
        "الجبيهة": (32.0167, 35.8833),
        "صويلح": (31.9833, 35.8333),
        "ناعور": (31.8667, 35.8167),
    },
    "إربد": {
        "إربد": (32.5556, 35.8500),
        "الرمثا": (32.5589, 36.0069),
        "الحصن": (32.4917, 35.8833),
        "بني عبيد": (32.5069, 35.7944),
    },
    "الزرقاء": {
        "الزرقاء": (32.0728, 36.0880),
        "الرصيفة": (32.0178, 36.0464),
        "الهاشمية": (32.1400, 36.0800),
    },
    "البلقاء": {
        "السلط": (32.0392, 35.7272),
        "الفحيص": (32.0000, 35.7833),
        "عين الباشا": (32.0833, 35.7500),
        "دير علا": (32.1833, 35.6000),
    },
    "مادبا": {
        "مادبا": (31.7167, 35.7939),
        "ذيبان": (31.5000, 35.7667),
        "ماعين": (31.6833, 35.7333),
    },
    "جرش": {
        "جرش": (32.2747, 35.8961),
        "برما": (32.3000, 35.7833),
        "سوف": (32.3667, 35.8833),
    },
    "عجلون": {
        "عجلون": (32.3333, 35.7500),
        "كفرنجة": (32.2667, 35.7500),
        "عنجرة": (32.2833, 35.7167),
    },
    "المفرق": {
        "المفرق": (32.3414, 36.2078),
        "رحاب": (32.2833, 36.1000),
        "البادية الشمالية": (32.4000, 36.3000),
    },
    "الكرك": {
        "الكرك": (31.1853, 35.7047),
        "المزار الجنوبي": (31.0667, 35.7000),
        "مؤتة": (31.0833, 35.7000),
    },
    "الطفيلة": {
        "الطفيلة": (30.8375, 35.6044),
        "بصيرا": (30.7333, 35.6167),
        "الحسا": (30.7833, 35.7500),
    },
    "معان": {
        "معان": (30.1962, 35.7345),
        "الشوبك": (30.5167, 35.5667),
        "وادي موسى": (30.3200, 35.4800),
    },
    "العقبة": {
        "العقبة": (29.5267, 35.0078),
        "وادي عربة": (29.8000, 35.1500),
        "الديسة": (29.6000, 35.4500),
    },
}


# ==========================================================
# 8. Crop Types and Advice
# ==========================================================
crop_types = {
    "🥬 خضروات ورقية": {
        "name": "الخضروات الورقية",
        "advice": (
            "تابعي درجات الحرارة ليلًا والصباح الباكر، "
            "وانتبهي للمحاصيل الحساسة للبرد."
        ),
    },
    "🍅 خضروات مثمرة": {
        "name": "الخضروات المثمرة",
        "advice": (
            "تابعي انخفاض درجات الحرارة واتخذي إجراءات "
            "الحماية المناسبة عند ارتفاع خطر الصقيع."
        ),
    },
    "🍊 أشجار مثمرة": {
        "name": "الأشجار المثمرة",
        "advice": (
            "راقبي فترة الصقيع المتوقعة، خصوصًا خلال "
            "ساعات الليل والصباح الباكر."
        ),
    },
    "🌾 محاصيل حقلية": {
        "name": "المحاصيل الحقلية",
        "advice": (
            "تابعي التوقعات الجوية وقيّمي حساسية المحصول "
            "للبرد قبل اتخاذ أي إجراء."
        ),
    },
    "🌿 أعشاب ونباتات": {
        "name": "الأعشاب والنباتات",
        "advice": (
            "تابعي درجات الحرارة واحمي النباتات الحساسة "
            "عند ارتفاع مستوى الخطر."
        ),
    },
}


# ==========================================================
# 9. Result HTML
# ==========================================================
def predict_frost_final(governorate, area, crop):
    try:
        if not governorate or not area or not crop:
            return (
                '<div class="error-card" dir="rtl">'
                "<h2>يرجى اختيار المحافظة والمنطقة والمحصول.</h2>"
                "</div>"
            )

        latitude, longitude = jordan_locations[governorate][area]
        crop_info = crop_types[crop]

        weather = check_frost_risk(latitude, longitude)

        min_temperature = float(
            weather["temperature_2m (°C)"].min()
        )

        probability = float(weather["frost_probability"].max())
        probability_percent = probability * 100

        if probability_percent >= 20:
            level = "🔴 خطر مرتفع"
            color = "#dc2626"
            background = "#fee2e2"
            risk_message = "توجد احتمالية مرتفعة للصقيع وفق الموديل."
            action = (
                "تابعي التحديثات الجوية واستشيري الإرشادات "
                "الزراعية المحلية بشأن حماية المحصول."
            )
        elif probability_percent >= 10:
            level = "🟡 خطر متوسط"
            color = "#d97706"
            background = "#fef3c7"
            risk_message = "توجد احتمالية متوسطة للصقيع وفق الموديل."
            action = "تابعي درجات الحرارة خلال الليل والصباح الباكر."
        else:
            level = "🟢 خطر منخفض"
            color = "#16a34a"
            background = "#dcfce7"
            risk_message = "احتمال الصقيع منخفض وفق توقعات الموديل الحالية."
            action = "استمري بمتابعة التحديثات الجوية."

        risk_hours = weather[
            weather["frost_probability"] >= threshold
        ]

        if not risk_hours.empty:
            first_time = str(risk_hours["time"].iloc[0])
            last_time = str(risk_hours["time"].iloc[-1])
            risk_period = f"من {first_time} إلى {last_time}"
        else:
            risk_period = (
                "لا توجد ساعات تجاوزت عتبة الخطر المحددة في الموديل."
            )

        hours_html = ""

        for _, row in weather.head(6).iterrows():
            temp = float(row["temperature_2m (°C)"])
            pct = float(row["frost_probability"]) * 100

            if pct >= 20:
                status = "🔴 مرتفع"
            elif pct >= 10:
                status = "🟡 متوسط"
            else:
                status = "🟢 منخفض"

            hours_html += f"""
                <tr>
                    <td>{row["time"]}</td>
                    <td>{temp:.1f} °C</td>
                    <td>{pct:.1f}%</td>
                    <td>{status}</td>
                </tr>
            """

        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

        return f"""
        <div class="result-card" dir="rtl">
            <h2 class="result-title">🌱 نتيجة فحص خطر الصقيع</h2>
            <p class="updated-at">وقت الفحص: {generated_at}</p>
            <hr class="green-line">

            <div class="info-section">
                <div class="info-label">📍 الموقع</div>
                <div class="info-value">{governorate} - {area}</div>
            </div>

            <div class="info-section">
                <div class="info-label">🌱 نوع المحصول</div>
                <div class="info-value crop-value">{crop_info['name']}</div>
            </div>

            <div class="info-section">
                <div class="info-label">🌡️ أقل درجة حرارة متوقعة</div>
                <div class="temperature-value">{min_temperature:.1f} °C</div>
            </div>

            <div class="info-section">
                <div class="info-label">❄️ أعلى احتمال صقيع يتنبأ به الموديل</div>
                <div class="temperature-value">{probability_percent:.1f}%</div>
            </div>

            <div class="info-section">
                <div class="info-label">⚠️ مستوى الخطر</div>
                <div class="risk-badge"
                     style="color:{color};background:{background};
                            border-color:{color}">
                    {level}
                </div>
            </div>

            <div class="info-section">
                <div class="info-label">🕐 فترة الخطر</div>
                <div class="risk-period">{risk_period}</div>
            </div>

            <div class="recommendation" style="background:{background}">
                <b>🌱 إرشادات للمحصول — {crop_info['name']}</b>
                <p>{risk_message}</p>
                <p>{action}</p>
                <p>{crop_info['advice']}</p>
            </div>

            <div class="forecast-section">
                <h3 class="forecast-title">🌤️ توقعات الساعات القادمة</h3>
                <div class="forecast-wrapper">
                    <table class="forecast-table">
                        <thead>
                            <tr>
                                <th>الوقت</th>
                                <th>الحرارة</th>
                                <th>احتمال الصقيع</th>
                                <th>الحالة</th>
                            </tr>
                        </thead>
                        <tbody>{hours_html}</tbody>
                    </table>
                </div>
            </div>

            <p class="disclaimer">
                النتيجة تقديرية وتعتمد على الموديل وبيانات الطقس المتاحة.
                احتمال الصقيع المعروض هو تقدير الموديل وليس بالضرورة
                احتمالًا رسميًا صادرًا عن خدمة الطقس.
            </p>
        </div>
        """

    except Exception as exc:
        print(f"[PREDICTION ERROR] {exc}", flush=True)

        error_text = (
            str(exc)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        return f"""
        <div class="error-card" dir="rtl">
            <h2>❌ تعذر فحص خطر الصقيع</h2>
            <p>{error_text}</p>
            <p>حاولي مرة أخرى لاحقًا إذا كانت خدمة الطقس غير متاحة.</p>
        </div>
        """


# ==========================================================
# 10. Update Areas
# ==========================================================
def update_area(governorate):
    areas = list(jordan_locations[governorate].keys())

    return gr.Dropdown(
        choices=areas,
        value=areas[0],
        label="📍 المنطقة",
    )


# ==========================================================
# 11. CSS
# ==========================================================
css = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
    background: linear-gradient(135deg, #052e16, #14532d, #166534) !important;
    min-height: 100vh;
}
body {
    background: #052e16 !important;
}
.header {
    background: linear-gradient(135deg, #166534, #15803d);
    color: white;
    text-align: center;
    padding: 30px;
    border-radius: 0 0 25px 25px;
    margin-bottom: 25px;
}
.header h1 {
    font-size: 32px !important;
}
.header p {
    font-size: 17px;
}
.location-box {
    background: white !important;
    padding: 25px;
    border-radius: 20px;
    border: 2px solid #22c55e !important;
}
#check-button {
    background: #f25c05 !important;
    color: white !important;
    border: 2px solid #111827 !important;
    height: 60px;
    border-radius: 14px;
    font-size: 21px !important;
    font-weight: bold;
}
.result-card {
    background: white !important;
    color: #172033 !important;
    padding: 30px;
    border-radius: 22px;
    margin-top: 25px;
    direction: rtl;
    text-align: right;
    font-family: Arial, sans-serif;
    box-shadow: 0 8px 30px rgba(0,0,0,.15);
}
.result-title {
    color: #166534 !important;
    text-align: center;
    font-size: 28px;
}
.updated-at {
    color: #64748b !important;
    text-align: center;
    font-size: 13px;
}
.green-line {
    border: 0;
    border-top: 2px solid #22c55e;
    margin: 20px 0;
}
.info-section {
    margin-top: 25px;
}
.info-label {
    color: #64748b !important;
    font-size: 17px;
}
.info-value {
    color: #111827 !important;
    font-size: 24px;
    font-weight: bold;
    margin-top: 5px;
}
.crop-value {
    color: #166534 !important;
}
.temperature-value {
    color: #111827 !important;
    font-size: 30px;
    font-weight: bold;
    margin-top: 5px;
}
.risk-badge {
    display: inline-block;
    margin-top: 8px;
    padding: 10px 25px;
    border-radius: 30px;
    font-size: 21px;
    font-weight: bold;
    border: 1px solid;
}
.risk-period {
    color: #111827 !important;
    font-size: 18px;
    font-weight: bold;
    margin-top: 8px;
}
.recommendation {
    margin-top: 30px;
    padding: 20px;
    border-radius: 15px;
    color: #334155 !important;
    font-size: 17px;
    line-height: 2;
}
.recommendation p {
    color: #334155 !important;
}
.recommendation b {
    color: #166534 !important;
    font-size: 20px;
}
.forecast-section {
    margin-top: 35px;
}
.forecast-title {
    color: #166534 !important;
    font-size: 23px;
}
.forecast-wrapper {
    overflow-x: auto;
    width: 100%;
}
.forecast-table {
    width: 100%;
    border-collapse: collapse;
    text-align: center;
    direction: rtl;
    background: white !important;
    font-size: 15px;
}
.forecast-table th {
    color: white !important;
    background: #166534 !important;
    padding: 12px;
}
.forecast-table td {
    color: #111827 !important;
    background: white !important;
    font-weight: 600;
    padding: 12px;
    border-bottom: 1px solid #d1d5db;
}
.disclaimer {
    margin-top: 25px;
    color: #64748b !important;
    font-size: 13px;
    line-height: 1.8;
}
.error-card {
    background: #fee2e2 !important;
    color: #991b1b !important;
    padding: 25px;
    border-radius: 15px;
    direction: rtl;
    text-align: right;
    margin-top: 20px;
}
.error-card h2,
.error-card p {
    color: #991b1b !important;
}
"""


# ==========================================================
# 12. Gradio Interface
# ==========================================================
with gr.Blocks(
    title="نظام الإنذار المبكر لخطر الصقيع",
    css=css,
) as app:

    gr.HTML("""
    <div class="header" dir="rtl">
        <h1>🇯🇴 نظام الإنذار المبكر لخطر الصقيع</h1>
        <p>نظام ذكي للتنبؤ بخطر الصقيع في الأردن</p>
        <p>اختاري موقعك لمعرفة مستوى الخطر المتوقع للمحصول.</p>
    </div>
    """)

    with gr.Column(elem_classes="location-box"):

        gr.HTML("""
        <div dir="rtl"
             style="text-align:right;padding-bottom:15px;
                    margin-bottom:20px;border-bottom:3px solid #22c55e">
            <h2 style="color:#166534;margin:0 0 8px 0;font-size:26px">
                📍 اختر موقعك
            </h2>
            <p style="color:#475569">
                حددي المحافظة والمنطقة والمحصول للحصول على نتيجة مخصصة.
            </p>
        </div>
        """)

        with gr.Row():
            governorate = gr.Dropdown(
                choices=list(jordan_locations.keys()),
                value="عمّان",
                label="📍 المحافظة",
            )

            area = gr.Dropdown(
                choices=list(jordan_locations["عمّان"].keys()),
                value="عمّان",
                label="📍 المنطقة",
            )

        crop = gr.Dropdown(
            choices=list(crop_types.keys()),
            value="🥬 خضروات ورقية",
            label="🌱 نوع المحصول",
        )

        check_button = gr.Button(
            "🔍 افحص خطر الصقيع",
            elem_id="check-button",
            variant="primary",
        )

    result = gr.HTML()

    governorate.change(
        fn=update_area,
        inputs=governorate,
        outputs=area,
    )

    check_button.click(
        fn=predict_frost_final,
        inputs=[governorate, area, crop],
        outputs=result,
    )


# ==========================================================
# 13. Start Server
# ==========================================================
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "10000")),
    )
