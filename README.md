# 🇯🇴 Jordan Frost Warning System

An AI-powered early warning system for frost risk in Jordan.

## 🌱 Project Overview

This project predicts frost risk based on weather conditions and provides a simple citizen-friendly interface.

Users can select:

- Governorate
- Area

The system automatically retrieves weather data and provides a frost risk assessment.

## 🤖 Machine Learning Model

The project uses a Random Forest Classifier.

The model uses the following weather features:

- Temperature
- Relative Humidity
- Dew Point
- Precipitation
- Cloud Cover
- Wind Speed

## 🌤️ Weather Data

Weather data is retrieved from the Open-Meteo API.

## 🚨 System Output

The system provides:

- Minimum expected temperature
- Frost probability
- Frost risk level
- Risk period
- Weather forecast for the upcoming hours
- Agricultural recommendations

Risk levels:

- 🟢 Low Risk
- 🟡 Medium Risk
- 🔴 High Risk

## 🗺️ Supported Locations

The system includes locations across Jordan, including:

- Amman
- Irbid
- Zarqa
- Balqa
- Madaba
- Jerash
- Ajloun
- Mafraq
- Karak
- Tafilah
- Ma'an
- Aqaba

## 🛠️ Technologies

- Python
- Pandas
- Scikit-learn
- Random Forest
- Gradio
- Open-Meteo API
- Render

## 🌐 Live Demo

https://jordan-frost-warning.onrender.com

## 📁 Project Files

- `app.py` — Main application
- `frost_random_forest.pkl` — Trained Random Forest model
- `frost_model_info.pkl` — Model configuration and features
- `requirements.txt` — Python dependencies

## 🚀 Deployment

The application is deployed using Render and provides a public web interface.

## 👩‍💻 Author

Areen Al Mahameed
<img width="1062" height="871" alt="Screenshot 2026-10-21 171450" src="https://github.com/user-attachments/assets/ca12ec3c-267c-4a82-ad17-8860543d33fc" />


