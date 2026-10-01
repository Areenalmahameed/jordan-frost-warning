import os
import joblib
import requests
import pandas as pd
import gradio as gr

model = joblib.load("frost_random_forest.pkl")
model_info = joblib.load("frost_model_info.pkl")
features = model_info["features"]
threshold = model_info.get("threshold", 0.20)


def get_weather(latitude, longitude):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "precipitation", "cloud_cover", "wind_speed_10m"
        ],
        "timezone": "Asia/Amman",
        "forecast_days": 3
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return pd.DataFrame(response.json()["hourly"])


def check_frost_risk(latitude, longitude):
    weather = get_weather(latitude, longitude).rename(columns={
        "temperature_2m": "temperature_2m (°C)",
        "relative_humidity_2m": "relative_humidity_2m (%)",
        "dew_point_2m": "dew_point_2m (°C)",
        "precipitation": "precipitation (mm)",
        "cloud_cover": "cloud_cover (%)",
        "wind_speed_10m": "wind_speed_10m (km/h)"
    }).copy()
    weather["frost_probability"] = model.predict_proba(weather[features])[:, 1]
    weather["frost_risk"] = (weather["frost_probability"] >= threshold).astype(int)
    return weather


jordan_locations = {
    "عمّان": {"عمّان": (31.9539, 35.9106), "ماركا": (31.9833, 35.9833), "الجبيهة": (32.0167, 35.8833), "صويلح": (32.0167, 35.8333), "ناعور": (31.8667, 35.8167)},
    "إربد": {"إربد": (32.5556, 35.8500), "الرمثا": (32.5589, 36.0069), "الحصن": (32.4917, 35.8833), "بني عبيد": (32.5069, 35.7944)},
    "الزرقاء": {"الزرقاء": (32.0728, 36.0880), "الرصيفة": (32.0178, 36.0464), "الهاشمية": (32.1400, 36.0800)},
    "البلقاء": {"السلط": (32.0392, 35.7272), "الفحيص": (32.0000, 35.7833), "عين الباشا": (32.0833, 35.7500), "دير علا": (32.1833, 35.6000)},
    "مادبا": {"مادبا": (31.7167, 35.7939), "ذيبان": (31.5000, 35.7667), "ماعين": (31.6833, 35.7333)},
    "جرش": {"جرش": (32.2747, 35.8961), "برما": (32.3000, 35.7833), "سوف": (32.3667, 35.8833)},
    "عجلون": {"عجلون": (32.3333, 35.7500), "كفرنجة": (32.2667, 35.7500), "عنجرة": (32.2833, 35.7167)},
    "المفرق": {"المفرق": (32.3414, 36.2078), "رحاب": (32.2833, 36.1000), "البادية الشمالية": (32.4000, 36.3000)},
    "الكرك": {"الكرك": (31.1853, 35.7047), "المزار الجنوبي": (31.0667, 35.7000), "مؤتة": (31.0833, 35.7000)},
    "الطفيلة": {"الطفيلة": (30.8375, 35.6044), "بصيرا": (30.7333, 35.6167), "الحسا": (30.7833, 35.7500)},
    "معان": {"معان": (30.1962, 35.7345), "الشوبك": (30.5167, 35.5667), "وادي موسى": (30.3200, 35.4800)},
    "العقبة": {"العقبة": (29.5267, 35.0078), "وادي عربة": (29.8000, 35.1500), "الديسة": (29.6000, 35.4500)}
}


def predict_frost_final(governorate, area):
    try:
        latitude, longitude = jordan_locations[governorate][area]
        weather = check_frost_risk(latitude, longitude).copy()
        min_temperature = weather["temperature_2m (°C)"].min()
        probability_percent = weather["frost_probability"].max() * 100

        if probability_percent >= 20:
            level, color, background = "🔴 خطر مرتفع", "#dc2626", "#fee2e2"
            recommendation = "<b>🚨 تنبيه مهم للمزارعين</b><br><br>توجد احتمالية مرتفعة للصقيع.<br>يُنصح بالاستعداد لحماية المحاصيل الحساسة واتخاذ الإجراءات الزراعية الوقائية المناسبة."
        elif probability_percent >= 10:
            level, color, background = "🟡 خطر متوسط", "#d97706", "#fef3c7"
            recommendation = "<b>⚠️ يُنصح بالاستعداد</b><br><br>توجد احتمالية للصقيع.<br>يُنصح بمتابعة درجات الحرارة خلال ساعات الليل والصباح الباكر والاستعداد إذا ارتفع مستوى الخطر."
        else:
            level, color, background = "🟢 خطر منخفض", "#16a34a", "#dcfce7"
            recommendation = "<b>✅ الحالة الحالية مطمئنة</b><br><br>لا يوجد خطر مرتفع للصقيع في الموقع المحدد حاليًا.<br>يُنصح بالاستمرار في متابعة التحديثات الجوية، خصوصًا خلال ساعات الليل والصباح الباكر."

        risk_hours = weather[weather["frost_probability"] >= threshold]
        if len(risk_hours):
            risk_period = f"من {risk_hours['time'].iloc[0]} إلى {risk_hours['time'].iloc[-1]}"
        else:
            risk_period = "لا توجد ساعات ذات خطر مرتفع حاليًا."

        rows = ""
        for _, row in weather.head(6).iterrows():
            p = row["frost_probability"] * 100
            status = "🔴 مرتفع" if p >= 20 else "🟡 متوسط" if p >= 10 else "🟢 منخفض"
            rows += f"<tr><td>{row['time']}</td><td>{row['temperature_2m (°C)']:.1f} °C</td><td>{p:.1f}%</td><td>{status}</td></tr>"

        return f"""
        <div class='result-card'>
          <h2>🌱 نتيجة فحص خطر الصقيع</h2><hr>
          <div class='info-section'><div class='label'>📍 الموقع</div><div class='value location-value'>{governorate} - {area}</div></div>
          <div class='info-section'><div class='label'>🌡️ أقل درجة حرارة متوقعة</div><div class='value big-value'>{min_temperature:.1f} °C</div></div>
          <div class='info-section'><div class='label'>❄️ احتمال الصقيع</div><div class='value big-value'>{probability_percent:.1f}%</div></div>
          <div class='info-section'><div class='label'>⚠️ مستوى الخطر</div><div class='risk-badge' style='color:{color};background:{background};border-color:{color};'>{level}</div></div>
          <div class='info-section'><div class='label'>🕐 فترة الخطر</div><div class='risk-period'>{risk_period}</div></div>
          <div class='recommendation' style='background:{background};'>{recommendation}</div>
          <div class='forecast-section'><h3>🌤️ توقعات الساعات القادمة</h3><div class='table-wrapper'>
            <table class='forecast-table'><thead><tr><th>الوقت</th><th>الحرارة</th><th>احتمال الصقيع</th><th>الحالة</th></tr></thead><tbody>{rows}</tbody></table>
          </div></div>
        </div>"""
    except Exception as e:
        return f"<div class='error-card'><h2>❌ حدث خطأ</h2><p>{e}</p></div>"


def update_area(governorate):
    areas = list(jordan_locations[governorate].keys())
    return gr.Dropdown(choices=areas, value=areas[0], label="📍 المنطقة")


css = """
.gradio-container{max-width:1000px!important;margin:auto!important;background:linear-gradient(135deg,#052e16 0%,#14532d 45%,#166534 100%)!important;min-height:100vh}
body{background:#052e16!important}.main{background:transparent!important}
.header{background:linear-gradient(135deg,#166534,#15803d);color:white;text-align:center;padding:30px;border-radius:0 0 25px 25px;margin-bottom:25px}.header h1{font-size:34px!important}.header p{font-size:17px}
.location-box{background:white!important;padding:25px;border-radius:20px;border:2px solid #22c55e!important}
#check-button{background:#f25c05!important;color:white!important;border:none!important;height:58px;border-radius:12px;font-size:20px!important;font-weight:bold}
.result-card{background:#fff;color:#172033;padding:30px;border-radius:22px;margin-top:20px;direction:rtl;text-align:right;font-family:Arial,sans-serif;box-shadow:0 8px 30px rgba(0,0,0,.15)}
.result-card h2{color:#166534;text-align:center;font-size:30px}.result-card hr{border:0;border-top:2px solid #22c55e;margin:20px 0}.info-section{margin-top:25px}.label{color:#64748b;font-size:17px}.value{color:#111827;font-weight:bold;margin-top:5px}.location-value{font-size:25px}.big-value{font-size:35px}
.risk-badge{display:inline-block;margin-top:8px;padding:10px 25px;border-radius:30px;font-size:23px;font-weight:bold;border:1px solid}.risk-period{color:#111827;font-size:20px;font-weight:bold;margin-top:8px}
.recommendation{margin-top:30px;padding:20px;border-radius:15px;color:#334155!important;font-size:18px;line-height:2}.recommendation p,.recommendation ul,.recommendation li,.recommendation span{color:#334155!important}.recommendation b{color:#166534!important;font-size:20px}
.forecast-section{margin-top:35px}.forecast-section h3{color:#166534;font-size:23px;margin-bottom:15px}.table-wrapper{overflow-x:auto}.forecast-table{width:100%;border-collapse:collapse;text-align:center;direction:rtl;background:#fff!important;border-radius:12px;overflow:hidden;font-size:16px}.forecast-table th{color:#fff!important;background:#166534!important;padding:12px;font-weight:bold}.forecast-table td{color:#111827!important;background:#fff!important;padding:12px;border-bottom:1px solid #e2e8f0;font-weight:600!important}.forecast-table td *{color:#111827!important}
.error-card{background:#fee2e2;color:#991b1b;padding:25px;border-radius:15px;direction:rtl;text-align:right;margin-top:20px}
"""

with gr.Blocks(title="🇯🇴 نظام الإنذار المبكر لخطر الصقيع", css=css) as app:
    gr.HTML("""
    <div class='header'><h1>🇯🇴 نظام الإنذار المبكر لخطر الصقيع</h1><p>نظام ذكي للتنبؤ بخطر الصقيع في الأردن</p><p>اختر موقعك في الأردن للحصول على تقييم خطر الصقيع بناءً على بيانات الطقس والنموذج التنبؤي.</p></div>
    """)
    with gr.Column(elem_classes="location-box"):
        gr.HTML("""
        <div style='direction:rtl;text-align:right;padding-bottom:15px;margin-bottom:20px;border-bottom:3px solid #22c55e'>
          <h2 style='color:#166534!important;margin:0 0 8px;font-size:26px;font-weight:700'>📍 اختر موقعك</h2>
          <p style='color:#475569!important;margin:0;font-size:16px;font-weight:500'>حدد المحافظة والمنطقة للحصول على نتيجة مخصصة لموقعك.</p>
        </div>
        """)
        with gr.Row():
            governorate = gr.Dropdown(choices=list(jordan_locations.keys()), value="عمّان", label="📍 المحافظة")
            area = gr.Dropdown(choices=list(jordan_locations["عمّان"].keys()), value="عمّان", label="📍 المنطقة")
        check_button = gr.Button("🔍 افحص خطر الصقيع", elem_id="check-button")
    result = gr.HTML()
    governorate.change(fn=update_area, inputs=governorate, outputs=area)
    check_button.click(fn=predict_frost_final, inputs=[governorate, area], outputs=result)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
