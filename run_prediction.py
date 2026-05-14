import os
import pickle
import requests
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from flask import Flask

# Inisialisasi Server Web Ringan
app = Flask(__name__)

# ================= KREDENSIAL KONFIGURASI =================
CHANNEL_ID = '3371118'
API_KEY = '2AAD8BIS31JN21GG'

BOT_TOKEN = '8213410722:AAEedN-gwEa0usOsdSq5o1qCydVtyoVmVp4'
CHAT_ID = '1364316606'

MODEL_PATH = 'model_lstm_pm25.keras'
SCALER_PATH = 'scaler_pm25.pkl'
window_size = 3

def eksekusi_prediksi_final():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("❌ File model.keras atau scaler.pkl belum ada!")
        return

    model = load_model(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    url_read = f'https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={API_KEY}&results=200'
    r = requests.get(url_read)
    data = r.json()

    if 'feeds' not in data or len(data['feeds']) == 0:
        print("❌ Gagal menarik data dari ThingSpeak.")
        return

    df = pd.DataFrame(data['feeds'])
    df['Dust'] = pd.to_numeric(df['field6'], errors='coerce')
    df['Time'] = pd.to_datetime(df['created_at'])
    df.set_index('Time', inplace=True)
    df.index = df.index.tz_convert('Asia/Jakarta')

    df_hourly = df[['Dust']].resample('1h').mean().ffill().dropna()

    if len(df_hourly) < window_size:
        print("❌ Data historis per jam kurang dari 3 jam.")
        return

    last_3_hours = df_hourly['Dust'].tail(window_size).values
    d_min2, d_min1, d_latest = last_3_hours[0], last_3_hours[1], last_3_hours[2]

    dummy_row = np.array([[0.0, d_latest, d_min1, d_min2]])
    scaled_dummy = scaler.transform(dummy_row)

    X_input = scaled_dummy[:, 1:].reshape(1, window_size, 1)
    pred_scaled = model.predict(X_input, verbose=0)

    combined = np.hstack((pred_scaled, X_input.reshape(1, window_size)))
    pred_actual = scaler.inverse_transform(combined)[0][0]
    
    if pred_actual < 0:
        pred_actual = 0.0

    # Eksekusi pengiriman output
    update_thingspeak_field7(pred_actual)
    kirim_notif_telegram(d_latest, pred_actual)
    
    return f"Sukses! Aktual: {d_latest:.1f}, Prediksi: {pred_actual:.1f}"

def update_thingspeak_field7(nilai_prediksi):
    url_write = f"https://api.thingspeak.com/update?api_key={API_KEY}&field7={nilai_prediksi:.2f}"
    requests.get(url_write)

def kirim_notif_telegram(aktual, prediksi):
    if prediksi > aktual:
        status_tren = "📈 Memburuk (Polusi Diprediksi Naik)"
    elif prediksi < aktual:
        status_tren = "📉 Membaik (Polusi Diprediksi Turun)"
    else:
        status_tren = "➡️ Stabil"

    pesan = (
        "🔮 *PREDIKSI KUALITAS UDARA (1 JAM KEDEPAN)* 🔮\n\n"
        f"📍 *Lokasi:* Balai Desa Gentan\n\n"
        f"💨 *PM2.5 Jam Terakhir:* {aktual:.1f} µg/m³\n"
        f"🎯 *Prediksi PM2.5:* *{prediksi:.1f} µg/m³*\n"
        f"📊 *Tren:* {status_tren}\n\n"
        "_Sistem diproses otomatis oleh model Deep Learning LSTM._"
    )

    url_tg = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url_tg, json={"chat_id": CHAT_ID, "text": pesan, "parse_mode": "Markdown"})

# --- ENDPOINT WEB UNTUK RENDER & CRON-JOB ---
@app.route('/')
def halaman_utama():
    return "✅ Server Machine Learning Gentan Aktif 24/7!"

@app.route('/jalankan-prediksi')
def trigger_prediksi():
    # URL ini yang nanti dipanggil otomatis tiap jam sama cron-job.org
    hasil = eksekusi_prediksi_final()
    return f"Proses Peramalan Selesai: {hasil}"

if __name__ == '__main__':
    # Render otomatis menyuntikkan environment port
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
