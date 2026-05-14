import os
import pickle
import requests
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# ================= KREDENSIAL KONFIGURASI =================
CHANNEL_ID = '3371118'
API_KEY = '2AAD8BIS31JN21GG'  # Berfungsi sebagai Read & Write Key

# Kredensial Bot Telegram
BOT_TOKEN = '8213410722:AAEedN-gwEa0usOsdSq5o1qCydVtyoVmVp4'
CHAT_ID = '1364316606'

# File Otak ML
MODEL_PATH = 'model_lstm_pm25.keras'
SCALER_PATH = 'scaler_pm25.pkl'
window_size = 3

def eksekusi_prediksi_final():
    # 1. Validasi keberadaan file model & scaler
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("❌ File model.keras atau scaler.pkl belum ada di direktori!")
        return

    print("🔄 Memuat model dan scaler matang...")
    model = load_model(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    # 2. Tarik data mentah secukupnya dari ThingSpeak (Ambil 200 baris terakhir)
    url_read = f'https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={API_KEY}&results=200'
    r = requests.get(url_read)
    data = r.json()

    if 'feeds' not in data or len(data['feeds']) == 0:
        print("❌ Gagal menarik data dari ThingSpeak.")
        return

    df = pd.DataFrame(data['feeds'])
    
    # Ambil kolom waktu dan field6 (Dust/PM2.5 mentah)
    df['Dust'] = pd.to_numeric(df['field6'], errors='coerce')
    df['Time'] = pd.to_datetime(df['created_at'])
    df.set_index('Time', inplace=True)
    df.index = df.index.tz_convert('Asia/Jakarta')

    # 3. Preprocessing: Resample rata-rata per jam biar deret waktunya rapi
    df_hourly = df[['Dust']].resample('1h').mean().ffill().dropna()

    # Pastikan minimal ada 3 jam data historis untuk sliding window
    if len(df_hourly) < window_size:
        print("❌ Data historis per jam yang valid masih kurang dari 3 jam.")
        return

    # Ambil nilai rata-rata PM2.5 di 3 jam terakhir
    # Urutan array: [jam_tertua, jam_tengah, jam_terbaru]
    last_3_hours = df_hourly['Dust'].tail(window_size).values
    d_min2, d_min1, d_latest = last_3_hours[0], last_3_hours[1], last_3_hours[2]

    # 4. Penyusunan Format Scaler (Krusial biar presisi)
    # Scaler di Colab dilatih pakai 4 kolom: [Dust_target, Dust_t-1, Dust_t-2, Dust_t-3]
    # Maka kita susun baris dummy dengan urutan fitur input yang sejajar
    dummy_row = np.array([[0.0, d_latest, d_min1, d_min2]])
    scaled_dummy = scaler.transform(dummy_row)

    # Ambil fitur X saja (kolom indeks 1, 2, 3) dan bentuk ke 3D tensor (1, 3, 1)
    X_input = scaled_dummy[:, 1:].reshape(1, window_size, 1)

    # 5. Lakukan Prediksi untuk 1 Jam ke Depan
    pred_scaled = model.predict(X_input, verbose=0)

    # Kembalikan skala ke satuan asli µg/m³
    combined = np.hstack((pred_scaled, X_input.reshape(1, window_size)))
    pred_actual = scaler.inverse_transform(combined)[0][0]
    
    # Pastikan nilai prediksi tidak minus
    if pred_actual < 0:
        pred_actual = 0.0

    print(f"✅ Hitungan ML Selesai!")
    print(f"   PM2.5 Jam Terakhir : {d_latest:.2f} µg/m³")
    print(f"   Prediksi Jam Depan : {pred_actual:.2f} µg/m³")

    # 6. Tulis Hasil Prediksi ke ThingSpeak Field 7
    update_thingspeak_field7(pred_actual)

    # 7. Kirim Notifikasi Prediksi ke Telegram
    kirim_notif_telegram(d_latest, pred_actual)


def update_thingspeak_field7(nilai_prediksi):
    # Nembak API ThingSpeak khusus untuk mengisi Field 7
    url_write = f"https://api.thingspeak.com/update?api_key={API_KEY}&field7={nilai_prediksi:.2f}"
    res = requests.get(url_write)
    if res.status_code == 200 and res.text != '0':
        print("✅ Data Prediksi sukses diunggah ke ThingSpeak Field 7!")
    else:
        print(f"⚠️ Gagal update ThingSpeak Field 7. Respon server: {res.text}")


def kirim_notif_telegram(aktual, prediksi):
    if prediksi > aktual:
        status_tren = "📈 Memburuk (Konsentrasi PM2.5 Diprediksi Naik)"
    elif prediksi < aktual:
        status_tren = "📉 Membaik (Konsentrasi PM2.5 Diprediksi Turun)"
    else:
        status_tren = "➡️ Stabil"

    pesan = (
        "🔮 *SISTEM PREDIKSI KUALITAS UDARA (LSTM)* 🔮\n\n"
        f"📍 *Lokasi:* Balai Desa Gentan\n\n"
        f"💨 *PM2.5 Rata-rata Jam Terakhir:* {aktual:.1f} µg/m³\n"
        f"🎯 *PREDIKSI PM2.5 (1 Jam Kedepan):* *{prediksi:.1f} µg/m³*\n"
        f"📊 *Tren Kondisi:* {status_tren}\n\n"
        "_Data dihitung secara nirkabel menggunakan Machine Learning._"
    )

    url_tg = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": pesan,
        "parse_mode": "Markdown"
    }
    
    res = requests.post(url_tg, json=payload)
    if res.status_code == 200:
        print("✅ Laporan hasil prediksi berhasil dikirim ke Telegram!")
    else:
        print("❌ Gagal mengirim pesan ke Telegram.")


if __name__ == '__main__':
    eksekusi_prediksi_final()