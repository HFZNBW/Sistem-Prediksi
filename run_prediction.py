import os
import pickle
import requests
import numpy as np
import pandas as pd
from flask import Flask
from tensorflow.keras.models import load_model

# --- INISIALISASI FLASK ---
app = Flask(__name__)

# --- KONFIGURASI KREDENSIAL ---
# Pastikan API_KEY ini bisa buat WRITE (nulis ke Field 7)
CHANNEL_ID = '3371118'
API_KEY = '2AAD8BIS31JN21GG' 
BOT_TOKEN = '8213410722:AAEedN-gwEa0usOsdSq5o1qCydVtyoVmVp4'
CHAT_ID = '1364316606'

# --- PATH FILE ---
MODEL_PATH = 'model_lstm_pm25.keras'
SCALER_PATH = 'scaler_pm25.pkl'

def proses_prediksi_gentan():
    print("--- Memulai Proses Prediksi ---")
    
    # 1. Load Model & Scaler
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("File model atau scaler tidak ditemukan!")
    
    model = load_model(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Model & Scaler loaded.")

    # 2. Tarik Data dari ThingSpeak (Field 6 = Dust)
    url_read = f'https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={API_KEY}&results=100'
    response = requests.get(url_read)
    data = response.json()
    
    if 'feeds' not in data or len(data['feeds']) < 10:
        return "Data di ThingSpeak belum cukup."

    df = pd.DataFrame(data['feeds'])
    df['Dust'] = pd.to_numeric(df['field6'], errors='coerce')
    df['Time'] = pd.to_datetime(df['created_at'])
    df.set_index('Time', inplace=True)
    # Convert ke WIB
    df.index = df.index.tz_convert('Asia/Jakarta')

    # 3. Resample Rata-rata per Jam
    df_hourly = df[['Dust']].resample('1h').mean().ffill().dropna()
    print(f"✅ Data processed. Jam terakhir: {df_hourly.index[-1]}")

    if len(df_hourly) < 3:
        return "Data historis per jam kurang (min 3 jam)."

    # 4. Ambil 3 Jam Terakhir untuk Input LSTM
    # Urutan: [t-3, t-2, t-1]
    last_3 = df_hourly['Dust'].tail(3).values
    d_min3, d_min2, d_latest = last_3[0], last_3[1], last_3[2]

    # Dummy untuk Scaler (4 kolom: Target, t-1, t-2, t-3)
    dummy = np.array([[0.0, d_latest, d_min2, d_min3]])
    scaled = scaler.transform(dummy)
    
    # Ambil kolom fitur (indeks 1, 2, 3) -> Reshape ke (1, 3, 1)
    X_input = scaled[:, 1:].reshape(1, 3, 1)
    
    # 5. Prediksi 1 Jam ke Depan
    pred_scaled = model.predict(X_input, verbose=0)
    
    # Invers Scaler
    combined = np.hstack((pred_scaled, X_input.reshape(1, 3)))
    pred_actual = scaler.inverse_transform(combined)[0][0]
    pred_actual = max(0, pred_actual) # Jangan biarkan negatif

    print(f"🎯 Hasil Prediksi: {pred_actual:.2f} µg/m³")

    # 6. Update ThingSpeak Field 7 (Data Prediksi)
    ts_write_url = f"https://api.thingspeak.com/update?api_key={API_KEY}&field7={pred_actual:.2f}"
    requests.get(ts_write_url)
    print("✅ Field 7 ThingSpeak Updated.")

    # 7. Kirim Notif Telegram
    tren = "📈 Memburuk" if pred_actual > d_latest else "📉 Membaik"
    pesan = (
        "🔮 *LAPORAN PREDIKSI UDARA GENTAN*\n\n"
        f"📍 Lokasi: Balai Desa\n"
        f"🕒 Jam Terakhir: {d_latest:.1f} µg/m³\n"
        f"🎯 *Prediksi 1 Jam Depan: {pred_actual:.1f} µg/m³*\n"
        f"📊 Tren: {tren}\n\n"
        "_Sistem berjalan otomatis tiap jam._"
    )
    
    tg_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(tg_url, json={"chat_id": CHAT_ID, "text": pesan, "parse_mode": "Markdown"})
    print("✅ Telegram Notification Sent.")

    return f"Prediksi Berhasil: {pred_actual:.2f}"

# --- ROUTES ---

@app.route('/')
def home():
    return "Sistem Prediksi Gentan Online. Gunakan endpoint /predict untuk eksekusi."

@app.route('/predict')
def predict():
    try:
        status = proses_prediksi_gentan()
        return {"status": "success", "message": status}, 200
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {"status": "error", "message": str(e)}, 500

if __name__ == "__main__":
    # Render mewajibkan host 0.0.0.0 dan port dinamis
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
