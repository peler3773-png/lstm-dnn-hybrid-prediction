import numpy as np
import pandas as pd
import requests
import json
from datetime import datetime, date, timedelta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# ==================================================
# 🔧 KONFIGURASI
# ==================================================

DATA_URL = "https://raw.githubusercontent.com/peler3773-png/Data-Lotre/main/data_undian.txt"

# Hanya kode yang diproses
KODE_DIPROSES = ["CLF", "BE", "SD", "SGP", "PS", "HK"]

PANJANG_URUTAN = 5
EPOCHS = 300
JAM_PREDIKSI = 20
PANJANG_ANGKA = 7  # ← 7 digit

pasaran_list = ["Legi", "Pahing", "Pon", "Wage", "Kliwon"]
neptu_pasaran = {"Legi":5, "Pahing":9, "Pon":7, "Wage":4, "Kliwon":8}
neptu_hari = {"Senin":4, "Selasa":3, "Rabu":7, "Kamis":8, "Jumat":6, "Sabtu":9, "Minggu":5}
hari_en_id = {
    "Monday":"Senin", "Tuesday":"Selasa", "Wednesday":"Rabu",
    "Thursday":"Kamis", "Friday":"Jumat", "Saturday":"Sabtu", "Sunday":"Minggu"
}

REF_TGL = date(2026, 10, 5)
REF_PASARAN = "Legi"
URUTAN_PASARAN = {"Legi":0, "Pahing":1, "Pon":2, "Wage":3, "Kliwon":4}

def hitung_pasaran(tgl):
    selisih = (tgl - REF_TGL).days
    return pasaran_list[(URUTAN_PASARAN[REF_PASARAN] + selisih) % 5]

# ==================================================
# 📥 AMBIL DATA
# ==================================================

print("📥 Mengunduh data...")
resp = requests.get(DATA_URL, timeout=60)
resp.raise_for_status()

baris = resp.text.strip().split("\n")
data_list = []
for b in baris:
    p = b.split("|")
    if len(p) < 4: continue
    kode = p[0].strip()
    if kode not in KODE_DIPROSES: continue
    tgl_str = p[1].strip()
    angka_str = p[2].strip()
    
    # Terima 4 atau 7 digit — sisanya isi nol
    if not angka_str.isdigit(): continue
    if len(angka_str) == 4:
        angka_str = angka_str + "000"  # 4 digit → tambah 000 jadi 7
    if len(angka_str) != PANJANG_ANGKA: continue
    
    try:
        tgl_obj = datetime.strptime(tgl_str, "%Y-%m-%d").date()
    except: continue
    hari = hari_en_id[tgl_obj.strftime("%A")]
    pasaran = hitung_pasaran(tgl_obj)
    data_list.append({
        "kode": kode, "tanggal": tgl_obj, "hari": hari,
        "pasaran": pasaran, "angka": int(angka_str),
        "neptu_h": neptu_hari[hari], "neptu_p": neptu_pasaran[pasaran]
    })

df_penuh = pd.DataFrame(data_list)
print(f"✅ Data dimuat — {len(df_penuh)} baris\n")

# ==================================================
# 🧠 PROSES SETIAP KODE
# ==================================================

hasil_akhir = []

for kode in KODE_DIPROSES:
    df = df_penuh[df_penuh["kode"] == kode].sort_values("tanggal").reset_index(drop=True)
    
    if len(df) < PANJANG_URUTAN + 3:
        print(f"⚠️ {kode} — Data tidak cukup ({len(df)} baris), dilewati")
        continue

    angka = df["angka"].values
    fitur = np.column_stack([
        df["pasaran"].map(lambda x: pasaran_list.index(x)).values,
        df["neptu_p"].values,
        df["neptu_h"].values,
        np.full(len(df), JAM_PREDIKSI)
    ])

    scaler_a = MinMaxScaler(feature_range=(0, 1))
    angka_s = scaler_a.fit_transform(angka.reshape(-1, 1))
    scaler_f = MinMaxScaler(feature_range=(0, 1))
    fitur_s = scaler_f.fit_transform(fitur)

    X_lstm, X_dnn, y = [], [], []
    for i in range(len(angka_s) - PANJANG_URUTAN):
        X_lstm.append(angka_s[i:i+PANJANG_URUTAN])
        X_dnn.append(fitur_s[i+PANJANG_URUTAN])
        y.append(angka_s[i+PANJANG_URUTAN])
    
    X_lstm = np.array(X_lstm).reshape(-1, PANJANG_URUTAN, 1)
    X_dnn = np.array(X_dnn)
    y = np.array(y)

    if len(X_lstm) < 4:
        print(f"⚠️ {kode} — Data terlalu sedikit, dilewati")
        continue

    batas = int(0.8 * len(X_lstm))
    X1t, X1v = X_lstm[:batas], X_lstm[batas:]
    X2t, X2v = X_dnn[:batas], X_dnn[batas:]
    yt, yv = y[:batas], y[batas:]

    # Model LSTM+DNN
    in1 = tf.keras.Input(shape=(PANJANG_URUTAN, 1))
    l1 = tf.keras.layers.LSTM(64, return_sequences=True)(in1)
    l1 = tf.keras.layers.LSTM(32)(l1)
    
    in2 = tf.keras.Input(shape=(4,))
    d1 = tf.keras.layers.Dense(32, activation='relu')(in2)
    
    gabung = tf.keras.layers.Concatenate()([l1, d1])
    out = tf.keras.layers.Dense(1)(gabung)
    
    model = tf.keras.Model([in1, in2], out)
    model.compile(optimizer="adam", loss="mse")

    model.fit([X1t, X2t], yt, epochs=EPOCHS, batch_size=2,
              validation_data=([X1v, X2v], yv), verbose=0)

    # Prediksi berikutnya
    terakhir = df.tail(PANJANG_URUTAN)["angka"].values
    tgl_terakhir = df["tanggal"].iloc[-1]
    tgl_nanti = tgl_terakhir + timedelta(days=1)
    hari_nanti = hari_en_id[tgl_nanti.strftime("%A")]
    pasaran_nanti = hitung_pasaran(tgl_nanti)

    inp1 = scaler_a.transform(terakhir.reshape(-1, 1)).reshape(1, PANJANG_URUTAN, 1)
    inp2 = scaler_f.transform([[
        pasaran_list.index(pasaran_nanti),
        neptu_pasaran[pasaran_nanti],
        neptu_hari[hari_nanti],
        JAM_PREDIKSI
    ]])

    pred = model.predict([inp1, inp2], verbose=0)
    angka_pred = int(round(scaler_a.inverse_transform(pred)[0][0]))
    
    # Format 7 digit
    angka_pred_str = f"{angka_pred:0{PANJANG_ANGKA}d}"
    angka_terakhir_str = f"{terakhir[-1]:0{PANJANG_ANGKA}d}"

    hasil_akhir.append({
        "kode": kode,
        "tanggal": tgl_nanti.strftime("%Y-%m-%d"),
        "hari": hari_nanti,
        "pasaran": pasaran_nanti,
        "neptu": f"{neptu_hari[hari_nanti]}+{neptu_pasaran[pasaran_nanti]}",
        "terakhir": angka_terakhir_str,
        "prediksi": angka_pred_str
    })

    print(f"✅ {kode} → {angka_pred_str}")

# ==================================================
# 💾 SIMPAN HASIL
# ==================================================

with open("hasil_prediksi.json", "w", encoding="utf-8") as f:
    json.dump(hasil_akhir, f, ensure_ascii=False, indent=2)

print(f"\n💾 Selesai! Diproses: {len(hasil_akhir)} dari {len(KODE_DIPROSES)} kode")
print(f"📄 Hasil disimpan ke: hasil_prediksi.json")
