import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pickle
import requests
import numpy as np
import pandas as pd
from flask import Flask
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# --- KONFIGURASI ---
CHANNEL_ID = '3371118'
API_KEY = '2AAD8BIS31JN21GG' 
BOT_TOKEN = '8213410722:AAEedN-gwEa0usOsdSq5o1qCydVtyoVmVp4'
CHAT_ID = '1364316606'
MODEL_PATH = 'model_lstm_pm25.h5' # Pastikan sudah pake .h5 biar aman

def get_ispu_info(val):
    """Fungsi pembantu klasifikasi status udara"""
    if val <= 15.5: return "🟢 BAIK"
    if val <= 55.4: return "🔵 SEDANG"
    if val <= 150.4: return "🟡 TIDAK SEHAT"
    if val <= 250.4: return "🔴 SANGAT TIDAK SEHAT"
    return "⚫ BERBAHAYA"

def jalankan_prediksi_total():
    tf.keras.backend.clear_session()
    model = load_model(MODEL_PATH)
    with open('scaler_pm25.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Ambil 50 data terakhir dari ThingSpeak
    url = f'https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={API_KEY}&results=50'
    data = requests.get(url).json()
    df = pd.DataFrame(data['feeds'])
    df['Dust'] = pd.to_numeric(df['field6'], errors='coerce')
    df = df.dropna(subset=['Dust'])

    if len(df) < 3: return "Data Kurang"

    # Data Aktual Terakhir
    d_now = df['Dust'].iloc[-1]
    last_3 = df['Dust'].tail(3).values

    # Proses Prediksi LSTM
    dummy = np.array([[0.0, last_3[2], last_3[1], last_3[0]]])
    scaled = scaler.transform(dummy)
    X_input = scaled[:, 1:].reshape(1, 3, 1)
    
    pred_scaled = model.predict(X_input, verbose=0)
    combined = np.hstack((pred_scaled, X_input.reshape(1, 3)))
    pred_val = max(0, scaler.inverse_transform(combined)[0][0])

    # 1. Update Field 7 ThingSpeak
    requests.get(f"https://api.thingspeak.com/update?api_key={API_KEY}&field7={pred_val:.2f}")

    # 2. Kirim Notif Telegram Terpadu
    msg = (
        "🔮 *SISTEM PREDIKSI UDARA GENTAN*\n\n"
        f"🕒 *KONDISI SAAT INI:*\n"
        f"PM2.5: {d_now:.1f} µg/m³\n"
        f"Status: {get_ispu_info(d_now)}\n\n"
        f"🎯 *RAMALAN 1 JAM KE DEPAN:*\n"
        f"PM2.5: {pred_val:.1f} µg/m³\n"
        f"Status: {get_ispu_info(pred_val)}\n\n"
        "_Update otomatis tiap 1 jam_"
    )
    
    requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", 
                  json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})
    
    return f"Success: {pred_val:.1f}"

@app.route('/predict')
def predict():
    try:
        res = jalankan_prediksi_total()
        return res, 200
    except Exception as e:
        return f"Error: {str(e)}", 200

@app.route('/')
def home():
    return "Server Active", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
