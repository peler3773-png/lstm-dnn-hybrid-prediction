
---

## 5. `model_hibrida.py` — Kode Utama
```python
import numpy as np
import pandas as pd
import requests
import json
from datetime import datetime, date, timedelta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import warnings
import os
warnings.filterwarnings("ignore")

# ==================================================
# 🔧 KONFIGURASI
# ==================================================

DATA_URL = "https://raw.githubusercontent.com/peler3773-png/Data-Lotre/main/data_undian.txt"
SIMPAN_FOLDER = "./model_simpan/"
os.makedirs(SIMPAN_FOLDER, exist_ok=True)

pasaran_list = ["Legi", "Pahing", "Pon", "Wage", "Kliwon"]
neptu_pasaran = {"Legi":5, "Pahing":9, "Pon":7, "Wage":4, "Kliwon":8}
neptu_hari = {"Senin":4, "Selasa":3, "Rabu":7, "Kamis":8, "Jumat":6, "Sabtu":9, "Minggu":5}
hari_inggris_ke_id = {
    "Monday":"Senin", "Tuesday":"Selasa", "Wednesday":"Rabu",
    "Thursday":"Kamis", "Friday":"Jumat", "Saturday":"Sabtu", "Sunday":"Minggu"
}

REF_TGL = date(2026, 10, 5)
REF_PASARAN = "Legi"
URUTAN_PASARAN = {"Legi":0, "Pahing":1, "Pon":2, "Wage":3, "Kliwon":4}

def hitung_pasaran(tanggal: date) -> str:
    selisih = (tanggal - REF_TGL).days
    idx = (URUTAN_PASARAN[REF_PASARAN] + selisih) % 5
    return pasaran_list[idx]

PANJANG_URUTAN = 5
EPOCHS = 600
BATCH_SIZE = 4
JAM_PREDIKSI = 20

KODE_DIPROSES = ["HK", "SGP", "TM", "TMD", "TXD", "TXE", "KYM", "FLM", "RIM", "INM", "MSM"]

# ==================================================
# 💾 FUNGSI SIMPAN & MUAT
# ==================================================

def dapatkan_nama_file(kode):
    return os.path.join(SIMPAN_FOLDER, f"model_{kode}.keras")

def dapatkan_nama_scaler(kode):
    return os.path.join(SIMPAN_FOLDER, f"scaler_{kode}.json")

def simpan_model(model, scaler_angka, scaler_fitur, kode):
    model.save(dapatkan_nama_file(kode))
    with open(dapatkan_nama_scaler(kode), "w") as f:
        json.dump({
            "scaler_angka_min": scaler_angka.min_.tolist(),
            "scaler_angka_scale": scaler_angka.scale_.tolist(),
            "scaler_fitur_min": scaler_fitur.min_.tolist(),
            "scaler_fitur_scale": scaler_fitur.scale_.tolist()
        }, f)
    print(f"✅ Model {kode} disimpan")

def muat_model(kode):
    jalur_model = dapatkan_nama_file(kode)
    jalur_scaler = dapatkan_nama_scaler(kode)
    if not os.path.exists(jalur_model) or not os.path.exists(jalur_scaler):
        return None, None, None
    model = tf.keras.models.load_model(jalur_model)
    with open(jalur_scaler, "r") as f:
        data = json.load(f)
    scaler_angka = MinMaxScaler()
    scaler_angka.min_ = np.array(data["scaler_angka_min"])
    scaler_angka.scale_ = np.array(data["scaler_angka_scale"])
    scaler_fitur = MinMaxScaler()
    scaler_fitur.min_ = np.array(data["scaler_fitur_min"])
    scaler_fitur.scale_ = np.array(data["scaler_fitur_scale"])
    print(f"✅ Model {kode} dimuat")
    return model, scaler_angka, scaler_fitur

# ==================================================
# 📥 AMBIL DATA
# ==================================================

print(f"📥 Mengunduh data...")
resp = requests.get(DATA_URL, timeout=30)
resp.raise_for_status()

baris = resp.text.strip().split("\n")
data_list = []
for b in baris:
    bagian = b.split("|")
    if len(bagian) < 4: continue
    kode = bagian[0].strip()
    tgl_str = bagian[1].strip()
    angka_str = bagian[2].strip()
    if len(angka_str) != 4 or not angka_str.isdigit(): continue
    try:
        tgl_obj = datetime.strptime(tgl_str, "%Y-%m-%d").date()
    except: continue
    nama_hari = hari_inggris_ke_id[tgl_obj.strftime("%A")]
    pasaran_nama = hitung_pasaran(tgl_obj)
    data_list.append({
        "kode": kode,
        "tanggal": tgl_obj,
        "hari": nama_hari,
        "pasaran": pasaran_nama,
        "angka": int(angka_str),
        "neptu_h": neptu_hari[nama_hari],
        "neptu_p": neptu_pasaran[pasaran_nama]
    })

df_penuh = pd.DataFrame(data_list)
print(f"✅ Data dimuat: {len(df_penuh)} baris\n")

# ==================================================
# 🧠 BANGUN MODEL
# ==================================================

def bangun_model(panjang_urutan):
    masukan_lstm = tf.keras.Input(shape=(panjang_urutan, 1))
    lstm_1 = tf.keras.layers.LSTM(128, return_sequences=True)(masukan_lstm)
    lstm_1 = tf.keras.layers.Dropout(0.25)(lstm_1)
    lstm_2 = tf.keras.layers.LSTM(64, return_sequences=False)(lstm_1)
    lstm_2 = tf.keras.layers.Dense(32, activation='relu')(lstm_2)

    masukan_dnn = tf.keras.Input(shape=(4,))
    dnn_1 = tf.keras.layers.Dense(64, activation='relu')(masukan_dnn)
    dnn_1 = tf.keras.layers.Dropout(0.2)(dnn_1)
    dnn_2 = tf.keras.layers.Dense(32, activation='relu')(dnn_1)

    gabung = tf.keras.layers.Concatenate()([lstm_2, dnn_2])
    lapisan_gabung = tf.keras.layers.Dense(64, activation='relu')(gabung)
    lapisan_gabung = tf.keras.layers.Dense(32, activation='relu')(lapisan_gabung)
    keluaran = tf.keras.layers.Dense(1, activation='linear')(lapisan_gabung)

    model = tf.keras.Model(inputs=[masukan_lstm, masukan_dnn], outputs=keluaran)
    model.compile(optimizer='adam', loss='mse')
    return model

# ==================================================
# 🔮 PROSES SEMUA KODE
# ==================================================

hasil_akhir = []

for kode in KODE_DIPROSES:
    print("-" * 50)
    print(f"Memproses: {kode}")

    df = df_penuh[df_penuh["kode"] == kode].sort_values("tanggal").reset_index(drop=True)
    if len(df) < PANJANG_URUTAN + 5:
        print(f"⚠️ Data kurang ({len(df)}) → dilewati")
        continue

    model, scaler_angka, scaler_fitur = muat_model(kode)

    if model is None:
        print("🔄 Melatih model baru...")
        angka_deret = df["angka"].values
        fitur_tambahan = np.column_stack([
            df["pasaran"].map(lambda x: pasaran_list.index(x)).values,
            df["neptu_p"].values,
            df["neptu_h"].values,
            np.full(len(df), JAM_PREDIKSI)
        ])

        scaler_angka = MinMaxScaler(feature_range=(0, 1))
        angka_scaled = scaler_angka.fit_transform(angka_deret.reshape(-1, 1))

        scaler_fitur = MinMaxScaler(feature_range=(0, 1))
        fitur_scaled = scaler_fitur.fit_transform(fitur_tambahan)

        X_lstm, X_dnn, y = [], [], []
        for i in range(len(angka_scaled) - PANJANG_URUTAN):
            X_lstm.append(angka_scaled[i:i+PANJANG_URUTAN])
            X_dnn.append(fitur_scaled[i+PANJANG_URUTAN])
            y.append(angka_scaled[i+PANJANG_URUTAN])
        X_lstm = np.array(X_lstm).reshape(-1, PANJANG_URUTAN, 1)
        X_dnn = np.array(X_dnn)
        y = np.array(y)

        batas = int(0.8 * len(X_lstm))
        X_lstm_train, X_lstm_test = X_lstm[:batas], X_lstm[batas:]
        X_dnn_train, X_dnn_test = X_dnn[:batas], X_dnn[batas:]
        y_train, y_test = y[:batas], y[batas:]

        model = bangun_model(PANJANG_URUTAN)
        model.fit(
            [X_lstm_train, X_dnn_train], y_train,
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            validation_data=([X_lstm_test, X_dnn_test], y_test),
            verbose=0
        )
        simpan_model(model, scaler_angka, scaler_fitur, kode)

    # Prediksi berikutnya
    data_terakhir = df.tail(PANJANG_URUTAN)
    angka_terbaru = data_terakhir["angka"].values
    tgl_terakhir = df["tanggal"].iloc[-1]

    tgl_nanti = tgl_terakhir + timedelta(days=1)
    nama_hari_nanti = hari_inggris_ke_id[tgl_nanti.strftime("%A")]
    pasaran_nanti = hitung_pasaran(tgl_nanti)

    lstm_input = scaler_angka.transform(angka_terbaru.reshape(-1,1)).reshape(1, PANJANG_URUTAN, 1)
    dnn_input = np.array([[
        pasaran_list.index(pasaran_nanti),
        neptu_pasaran[pasaran_nanti],
        neptu_hari[nama_hari_nanti],
        JAM_PREDIKSI
    ]])
    dnn_input = scaler_fitur.transform(dnn_input)

    hasil = model.predict([lstm_input, dnn_input], verbose=0)
    angka_prediksi = int(round(scaler_angka.inverse_transform(hasil)[0][0]))

    hasil_akhir.append({
        "Kode": kode,
        "Tanggal": tgl_nanti.strftime("%Y-%m-%d"),
        "Hari": nama_hari_nanti,
        "Pasaran": pasaran_nanti,
        "Neptu": f"{neptu_hari[nama_hari_nanti]}+{neptu_pasaran[pasaran_nanti]}",
        "Terakhir": f"{angka_terbaru[-1]:04d}",
        "Prediksi": f"{angka_prediksi:04d}"
    })

# ==================================================
# 📋 TABEL HASIL
# ==================================================

print("\n" + "="*70)
print("📊 RINGKASAN SEMUA PREDIKSI")
print("="*70)
print(f"{'Kode':<8} | {'Tanggal':<12} | {'Hari':<8} | {'Pasaran':<8} | {'Neptu':<7} | {'Terakhir':<9} | {'Prediksi':<8}")
print("-"*70)
for h in hasil_akhir:
    print(f"{h['Kode']:<8} | {h['Tanggal']:<12} | {h['Hari']:<8} | {h['Pasaran']:<8} | {h['Neptu']:<7} | {h['Terakhir']:<9} | {h['Prediksi']:<8}")
print("="*70)
print(f"\n✅ Selesai! Diproses: {len(hasil_akhir)} kode")
print(f"💾 Model tersimpan di: {SIMPAN_FOLDER}")
