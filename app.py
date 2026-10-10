
# -*- coding: utf-8 -*-
import os
import json
import time
from pathlib import Path
from datetime import datetime

import gradio as gr
import joblib
import pandas as pd
import requests


# ==========================================================
# 1. Load Model
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "frost_random_forest.pkl"
INFO_PATH = BASE_DIR / "frost_model_info.pkl"
CACHE_FILE = BASE_DIR / "weather_cache.json"

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

CACHE_DURATION = 6 * 60 * 60
MAX_STALE_DURATION = 48 * 60 * 60

if not MODEL_PATH.exists() or not INFO_PATH.exists():
    raise FileNotFoundError(
        "تأكدي من وجود frost_random_forest.pkl "
        "و frost_model_info.pkl بجانب app.py"
    )

model = joblib.load(MODEL_PATH)
model_info = joblib.load(INFO_PATH)

features = model_info["features"]
threshold = float(model_info.get("threshold", 0.20))


# ==========================================================
# 2. Persistent Weather Cache
# ==========================================================
def load_cache():
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)
                return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        pass

    return {}


weather_cache = load_cache()


def save_cache():
    """Save successful forecasts to disk."""
    temp_file = CACHE_FILE.with_suffix(".tmp")

    try:
        with open(temp_file, "w", encoding="utf-8") as file:
            json.dump(weather_cache, file, ensure_ascii=False)

        os.replace(temp_file, CACHE_FILE)

    except (OSError, TypeError, ValueError) as exc:
        print(f"[CACHE] Could not save cache: {exc}", flush=True)


def cache_key_for(latitude, longitude):
    return f"{float(latitude):.4f},{float(longitude):.4f}"


def get_cached_weather(key):
    entry = weather_cache.get(key)

    if not isinstance(entry, dict):
        return None, None

    try:
        timestamp = float(entry["timestamp"])
        rows = entry["data"]
        weather = pd.DataFrame(rows)

        if weather.empty:
            return None, None

        return timestamp, weather

    except (KeyError, TypeError, ValueError):
        return None, None


# ==========================================================
# 3. Get Weather
# ==========================================================
def get_weather(latitude, longitude):
    latitude = float(latitude)
    longitude = float(longitude)

    key = cache_key_for(latitude, longitude)
    now = time.time()

    cached_timestamp, cached_weather = get_cached_weather(key)

    # Return fresh cached data without contacting the API.
    if (
        cached_weather is not None
        and cached_timestamp is not None
        and 0 <= now - cached_timestamp < CACHE_DURATION
    ):
        print(f"[CACHE] Fresh cache used for {key}", flush=True)
        return cached_weather.copy()

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

    last_error = "Unknown error"

    # Make one request per attempt; avoid excessive API requests.
    try:
        response = requests.get(
            WEATHER_URL,
            params=params,
            timeout=20,
            headers={"User-Agent": "JordanFrostWarning/1.0"},
        )

        print(
            f"[WEATHER] HTTP status: {response.status_code}",
            flush=True,
        )

        if response.status_code == 429:
            last_error = "Open-Meteo HTTP 429: Too Many Requests"

        else:
            response.raise_for_status()
            payload = response.json()

            if "hourly" not in payload:
                raise ValueError(
                    "لم تُرجع خدمة الطقس بيانات الساعات المطلوبة."
                )

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

            missing = [
                column for column in required
                if column not in weather.columns
            ]

            if missing:
                raise ValueError(
                    f"Missing weather fields: {missing}"
                )

            if weather.empty:
                raise ValueError("بيانات الطقس فارغة.")

            # Save the forecast and its retrieval timestamp.
            weather_cache[key] = {
                "timestamp": time.time(),
                "data": weather.to_dict(orient="records"),
            }
            save_cache()

            return weather

    except requests.exceptions.RequestException as exc:
        last_error = str(exc)

    except (ValueError, KeyError, TypeError) as exc:
        last_error = str(exc)

    # If the API fails, use a previously saved forecast if recent enough.
    cached_timestamp, cached_weather = get_cached_weather(key)

    if cached_weather is not None and cached_timestamp is not None:
        age = time.time() - cached_timestamp

        if 0 <= age <= MAX_STALE_DURATION:
            print(
                f"[CACHE] Using saved forecast for {key}; "
                f"age={age / 3600:.1f} hours",
                flush=True,
            )
            return cached_weather.copy()

    raise RuntimeError(
        "تعذر جلب بيانات الطقس حاليًا، ولا توجد بيانات محفوظة "
        "صالحة لهذا الموقع. انتظري قليلًا ثم حاولي مرة أخرى. "
        f"Details: {last_error}"
    )


# ==========================================================
# 4. Frost Risk Prediction
# ==========================================================
def check_frost_risk(latitude, longitude):
    weather = get_weather(latitude, longitude)

    # Match API column names to the model's expected feature names.
    rename_map = {
        "temperature_2m": "temperature_2m (°C)",
        "relative_humidity_2m": "relative_humidity_2m (%)",
        "dew_point_2m": "dew_point_2m (°C)",
        "precipitation": "precipitation (mm)",
        "cloud_cover": "cloud_cover (%)",
        "wind_speed_10m": "wind_speed_10m (km/h)",
    }

    weather = weather.rename(columns=rename_map).copy()

    missing = [feature for feature in features
               if feature not in weather.columns]

    if missing:
        raise ValueError(
            "خصائص الطقس لا تتطابق مع الموديل. "
            f"الخصائص المفقودة: {missing}"
        )

    # Keep the exact feature order used during training.
    X = weather[features].copy()
    probabilities = model.predict_proba(X)

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            "الموديل لا يحتوي على الفئة 1 الخاصة بالصقيع. "
            "راجعي ترميز الفئات عند التدريب."
        )

    frost_index = classes.index(1)
    weather["frost_probability"] = probabilities[:, frost_index]
    weather["frost_risk"] = (
        weather["frost_probability"] >= threshold
    ).astype(int)

    return weather


# ==========================================================
# 5. Jordan Locations
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
# 6. Crop Types and Advice
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
# 7. Generate Result
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

        weather = check_frost_risk(latitude, longitude).copy()

        min_temperature = float(
            weather["temperature_2m (°C)"].min()
        )

        # Use the highest predicted probability in the forecast.
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
            action = (
                "تابعي درجات الحرارة خلال الليل والصباح الباكر."
            )

        else:
            level = "🟢 خطر منخفض"
            color = "#16a34a"
            background = "#dcfce7"
            risk_message = (
                "احتمال الصقيع منخفض وفق توقعات الموديل الحالية."
            )
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

        # Display the next six forecast entries.
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

            forecast_time = str(row["time"])

            hours_html += f"""
                <tr>
                    <td>{forecast_time}</td>
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
                النتيجة تقديرية وتعتمد على الموديل المدرب وبيانات الطقس.
                احتمال الصقيع الذي يعرضه التطبيق هو تقدير الموديل،
                وليس بالضرورة احتمالًا رسميًا صادرًا عن خدمة الطقس.
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
# 8. Update Areas When Governorate Changes
# ==========================================================
def update_area(governorate):
    areas = list(jordan_locations[governorate].keys())

    return gr.Dropdown(
        choices=areas,
        value=areas[0],
        label="📍 المنطقة",
    )


# ==========================================================
# 9. CSS Design
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
# 10. Gradio Interface
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
# 11. Start Server
# ==========================================================
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "10000")),
    )
