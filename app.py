# -*- coding: utf-8 -*-
import os
import time
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd
import requests

# ==========================================================
# Load Model
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "frost_random_forest.pkl"
INFO_PATH = BASE_DIR / "frost_model_info.pkl"

if not MODEL_PATH.exists() or not INFO_PATH.exists():
    raise FileNotFoundError(
        "تأكدي من وجود frost_random_forest.pkl و frost_model_info.pkl"
    )

model = joblib.load(MODEL_PATH)
model_info = joblib.load(INFO_PATH)
features = model_info["features"]
threshold = float(model_info.get("threshold", 0.20))

# ==========================================================
# Weather API + Cache
# ==========================================================
weather_cache = {}
CACHE_DURATION = 30 * 60
MAX_STALE_DURATION = 6 * 60 * 60
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


def get_weather(latitude, longitude):
    cache_key = (round(float(latitude), 4), round(float(longitude), 4))
    now = time.time()
    cached = weather_cache.get(cache_key)

    if cached and now - cached[0] < CACHE_DURATION:
        return cached[1].copy()

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "precipitation",
            "cloud_cover",
            "wind_speed_10m",
        ],
        "timezone": "Asia/Amman",
        "forecast_days": 3,
    }

    last_error = "Unknown error"

    for attempt in range(3):
        try:
            response = requests.get(
                WEATHER_URL,
                params=params,
                timeout=20,
            )

            if response.status_code == 429:
                last_error = "Open-Meteo HTTP 429: Too Many Requests"

                if attempt < 2:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        wait_seconds = max(1, float(retry_after))
                    except (TypeError, ValueError):
                        wait_seconds = 5 * (2 ** attempt)

                    time.sleep(min(wait_seconds, 20))
                    continue
                break

            response.raise_for_status()
            payload = response.json()

            if "hourly" not in payload:
                raise ValueError("Weather API did not return hourly data")

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

            missing = [c for c in required if c not in weather.columns]
            if missing:
                raise ValueError(f"Missing weather fields: {missing}")

            if weather.empty:
                raise ValueError("Weather API returned an empty forecast")

            weather_cache[cache_key] = (time.time(), weather.copy())
            return weather

        except requests.exceptions.RequestException as exc:
            last_error = str(exc)
            if attempt < 2:
                time.sleep(2 * (2 ** attempt))

        except (ValueError, KeyError) as exc:
            last_error = str(exc)
            break

    # Use cached data only when a previous successful response exists.
    if cached and time.time() - cached[0] <= MAX_STALE_DURATION:
        return cached[1].copy()

    raise RuntimeError(
        "تعذر جلب بيانات الطقس حاليًا، ولا توجد بيانات محفوظة "
        "حديثة بما يكفي. حاولي مرة أخرى لاحقًا. "
        f"Details: {last_error}"
    )


def check_frost_risk(latitude, longitude):
    weather = get_weather(latitude, longitude).rename(columns={
        "temperature_2m": "temperature_2m (°C)",
        "relative_humidity_2m": "relative_humidity_2m (%)",
        "dew_point_2m": "dew_point_2m (°C)",
        "precipitation": "precipitation (mm)",
        "cloud_cover": "cloud_cover (%)",
        "wind_speed_10m": "wind_speed_10m (km/h)",
    }).copy()

    missing = [f for f in features if f not in weather.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    probabilities = model.predict_proba(weather[features])
    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("The model does not contain frost class 1")

    weather["frost_probability"] = probabilities[:, classes.index(1)]
    weather["frost_risk"] = (
        weather["frost_probability"] >= threshold
    ).astype(int)

    return weather


# ==========================================================
# Jordan Locations
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
# Crop Types
# ==========================================================
crop_types = {
    "🥬 خضروات ورقية": {
        "name": "الخضروات الورقية",
        "advice": "تابعي درجات الحرارة ليلًا والصباح الباكر، وانتَبهي للمحاصيل الحساسة للبرد.",
    },
    "🍅 خضروات مثمرة": {
        "name": "الخضروات المثمرة",
        "advice": "تابعي انخفاض درجات الحرارة واتخذي إجراءات الحماية المناسبة عند ارتفاع خطر الصقيع.",
    },
    "🍊 أشجار مثمرة": {
        "name": "الأشجار المثمرة",
        "advice": "راقبي فترة الصقيع المتوقعة، خصوصًا خلال ساعات الليل والصباح الباكر.",
    },
    "🌾 محاصيل حقلية": {
        "name": "المحاصيل الحقلية",
        "advice": "تابعي التوقعات الجوية وقيّمي حساسية المحصول للبرد قبل اتخاذ أي إجراء.",
    },
    "🌿 أعشاب ونباتات": {
        "name": "الأعشاب والنباتات",
        "advice": "تابعي درجات الحرارة واحمي النباتات الحساسة عند ارتفاع مستوى الخطر.",
    },
}


# ==========================================================
# Prediction
# ==========================================================
def predict_frost_final(governorate, area, crop):
    try:
        latitude, longitude = jordan_locations[governorate][area]
        crop_info = crop_types[crop]

        weather = check_frost_risk(latitude, longitude).copy()

        min_temperature = weather["temperature_2m (°C)"].min()
        probability = weather["frost_probability"].max()
        probability_percent = probability * 100

        if probability_percent >= 20:
            level = "🔴 خطر مرتفع"
            color = "#dc2626"
            background = "#fee2e2"
            risk_message = "توجد احتمالية مرتفعة للصقيع."
            action = "استعدي لحماية المحصول واتخذي الإجراءات الوقائية المناسبة."
        elif probability_percent >= 10:
            level = "🟡 خطر متوسط"
            color = "#d97706"
            background = "#fef3c7"
            risk_message = "توجد احتمالية متوسطة للصقيع."
            action = "تابعي درجات الحرارة خلال الليل والصباح الباكر."
        else:
            level = "🟢 خطر منخفض"
            color = "#16a34a"
            background = "#dcfce7"
            risk_message = "لا يوجد خطر مرتفع للصقيع وفق التوقعات الحالية."
            action = "استمري بمتابعة التحديثات الجوية."

        risk_hours = weather[weather["frost_probability"] >= 0.20]

        if not risk_hours.empty:
            risk_period = (
                f"من {risk_hours['time'].iloc[0]} "
                f"إلى {risk_hours['time'].iloc[-1]}"
            )
        else:
            risk_period = "لا توجد ساعات ذات خطر مرتفع وفق العتبة المحددة."

        hours_html = ""

        for _, row in weather.head(6).iterrows():
            temp = row["temperature_2m (°C)"]
            pct = row["frost_probability"] * 100

            if pct >= 20:
                status = "🔴 مرتفع"
            elif pct >= 10:
                status = "🟡 متوسط"
            else:
                status = "🟢 منخفض"

            hours_html += f"""
            <tr>
                <td>{row['time']}</td>
                <td>{temp:.1f} °C</td>
                <td>{pct:.1f}%</td>
                <td>{status}</td>
            </tr>
            """

        return f"""
        <div class="result-card" dir="rtl">
            <h2 class="result-title">🌱 نتيجة فحص خطر الصقيع</h2>
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
                <div class="info-label">❄️ احتمال الصقيع</div>
                <div class="temperature-value">{probability_percent:.1f}%</div>
            </div>

            <div class="info-section">
                <div class="info-label">⚠️ مستوى الخطر</div>
                <div class="risk-badge"
                     style="color:{color};background:{background};border-color:{color}">
                    {level}
                </div>
            </div>

            <div class="info-section">
                <div class="info-label">🕐 فترة الخطر</div>
                <div class="risk-period">{risk_period}</div>
            </div>

            <div class="recommendation" style="background:{background}">
                <b>🌱 توصيات للمزارع — {crop_info['name']}</b>
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
                هذه نتيجة تقديرية تعتمد على بيانات الطقس والموديل المدرب،
                وليست ضمانًا لحدوث الصقيع أو عدم حدوثه.
            </p>
        </div>
        """

    except Exception as exc:
        # Avoid showing a technical traceback to the app visitor.
        error_text = str(exc).replace("<", "&lt;").replace(">", "&gt;")
        return f"""
        <div class="error-card" dir="rtl">
            <h2>❌ تعذر فحص خطر الصقيع</h2>
            <p>{error_text}</p>
            <p>انتظري قليلًا ثم حاولي مرة أخرى.</p>
        </div>
        """


def update_area(governorate):
    areas = list(jordan_locations[governorate].keys())
    return gr.Dropdown(
        choices=areas,
        value=areas[0],
        label="📍 المنطقة",
    )


# ==========================================================
# CSS
# ==========================================================
css = """
.gradio-container {
    max-width: 1000px !important;
    margin: auto !important;
    background: linear-gradient(135deg, #052e16, #14532d, #166534) !important;
    min-height: 100vh;
}
body { background: #052e16 !important; }

.header {
    background: linear-gradient(135deg, #166534, #15803d);
    color: white;
    text-align: center;
    padding: 30px;
    border-radius: 0 0 25px 25px;
    margin-bottom: 25px;
}
.header h1 { font-size: 32px !important; }

.location-box {
    background: white !important;
    padding: 25px;
    border-radius: 20px;
    border: 2px solid #22c55e !important;
}

#check-button {
    background: #f25c05 !important;
    color: white !important;
    border: none !important;
    height: 58px;
    border-radius: 12px;
    font-size: 20px !important;
    font-weight: bold;
}

.result-card {
    background: white !important;
    color: #172033 !important;
    padding: 30px;
    border-radius: 22px;
    margin-top: 20px;
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
.green-line {
    border: 0;
    border-top: 2px solid #22c55e;
    margin: 20px 0;
}

.info-section { margin-top: 25px; }
.info-label { color: #64748b !important; font-size: 17px; }
.info-value {
    color: #111827 !important;
    font-size: 25px;
    font-weight: bold;
}
.crop-value { color: #166534 !important; }
.temperature-value {
    color: #111827 !important;
    font-size: 32px;
    font-weight: bold;
}

.risk-badge {
    display: inline-block;
    margin-top: 8px;
    padding: 10px 25px;
    border-radius: 30px;
    font-size: 22px;
    font-weight: bold;
    border: 1px solid;
}
.risk-period {
    color: #111827 !important;
    font-size: 20px;
    font-weight: bold;
    margin-top: 8px;
}

.recommendation {
    margin-top: 30px;
    padding: 20px;
    border-radius: 15px;
    color: #334155 !important;
    font-size: 18px;
    line-height: 2;
}
.recommendation p,
.recommendation b {
    color: #334155 !important;
}
.recommendation b {
    color: #166534 !important;
    font-size: 20px;
}

.forecast-section { margin-top: 35px; }
.forecast-title {
    color: #166534 !important;
    font-size: 23px;
}
.forecast-wrapper { overflow-x: auto; width: 100%; }
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
.error-card h2, .error-card p { color: #991b1b !important; }
"""


# ==========================================================
# Gradio Interface
# ==========================================================
with gr.Blocks(
    title="🇯🇴 نظام الإنذار المبكر لخطر الصقيع",
    css=css,
) as app:

    gr.HTML("""
    <div class="header">
        <h1>🇯🇴 نظام الإنذار المبكر لخطر الصقيع</h1>
        <p>نظام ذكي للتنبؤ بخطر الصقيع في الأردن</p>
        <p>اختر موقعك في الأردن لمعرفة مستوى خطر الصقيع المتوقع.</p>
    </div>
    """)

    with gr.Column(elem_classes="location-box"):
        gr.HTML("""
        <div dir="rtl" style="text-align:right;padding-bottom:15px;
             margin-bottom:20px;border-bottom:3px solid #22c55e">
            <h2 style="color:#166534;margin:0 0 8px 0;font-size:26px">
                📍 اختر موقعك
            </h2>
            <p style="color:#475569">
                حدد المحافظة والمنطقة والمحصول للحصول على نتيجة مخصصة.
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


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        share=False,
    )
