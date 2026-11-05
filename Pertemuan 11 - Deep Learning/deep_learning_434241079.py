# =============================================================
#  TUGAS DEEP LEARNING - MACHINE LEARNING (Pertemuan 11)
#  File    : deep_learning_434241079.py
#  Nama    : Muhammad Irfan Nuha  |  NIM : 434241079  |  TI-C5
#  Dataset : Bla Bla Bla (BlaBla.xlsx)
#  Model   : Jaringan saraf tiruan (Keras Sequential) untuk
#            klasifikasi biner kolom N
# =============================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"        # sembunyikan log internal TensorFlow
import warnings
warnings.filterwarnings("ignore")

# 1. Import Library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

tf.keras.utils.set_random_seed(42)               # agar hasil training bisa diulang

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)


def judul(teks):
    print("\n" + "=" * 62)
    print(f"  {teks}")
    print("=" * 62)


def simpan_dan_tampilkan(nama_file):
    plt.savefig(os.path.join(OUT_DIR, nama_file), dpi=110, bbox_inches="tight")
    plt.show()
    plt.close()


# 2. Load dataset
judul("2. LOAD DATASET")
df = pd.read_excel(os.path.join(BASE_DIR, "BlaBla.xlsx"))
print(f"Ukuran data : {df.shape[0]} baris x {df.shape[1]} kolom")
print(df.head().to_string(index=False))

# 3. Pisahkan fitur dan label
judul("3. PISAHKAN FITUR DAN LABEL")
FITUR = ["Umur"] + list("BCDEFGHIJKLM")     # variabel input : Umur, B sampai M
LABEL = "N"                                  # variabel output: N
print(f"   Input  ({len(FITUR)} fitur) : {FITUR}")
print(f"   Output           : {LABEL}  ->  {df[LABEL].value_counts().sort_index().to_dict()}")

# 4. Lakukan Encoding -> Umur
judul("4. ENCODING UMUR")
umur_tahun = pd.to_numeric(df["UMUR_TAHUN"], errors="coerce")   # teks salah ketik -> NaN


def kode_umur(u):
    if pd.isna(u):
        return np.nan
    if u <= 20:
        return 1          # <= 20
    if u <= 30:
        return 2          # 21 - 30
    if u <= 40:
        return 3          # 31 - 40
    if u <= 50:
        return 4          # 41 - 50
    return 5              # > 50


df["Umur"] = umur_tahun.apply(kode_umur)
print("   Aturan: 1 = <=20 | 2 = 21-30 | 3 = 31-40 | 4 = 41-50 | 5 = >50")
print(f"   Sebaran kode Umur: {df['Umur'].value_counts().sort_index().to_dict()}")
sama = (df["Umur"].dropna() == df.loc[df["Umur"].notna(), "A"]).mean() * 100
print(f"   Kecocokan dengan kolom A pada data asli: {sama:.1f}%")

# 5. Lakukan Preprocessing Data
judul("5. PREPROCESSING DATA")
# a. Cek Missing Value
print(">> a. Missing value:")
salah = df.loc[umur_tahun.isna(), "UMUR_TAHUN"].tolist()
print(f"     UMUR_TAHUN bukan angka : {len(salah)} baris {salah}")
print(f"     Missing pada fitur      : {df[FITUR].isna().sum()[df[FITUR].isna().sum() > 0].to_dict()}")
df["Umur"] = df["Umur"].fillna(df["A"])      # isi dari kode umur kolom A (data asli)
print(f"     -> diisi dari kolom A, sisa missing = {int(df[FITUR + [LABEL]].isna().sum().sum())}")

# b. Outlier
print(">> b. Outlier:")
q1, q3 = umur_tahun.quantile(0.25), umur_tahun.quantile(0.75)
iqr = q3 - q1
bawah, atas = q1 - 1.5 * iqr, q3 + 1.5 * iqr
n_out = int(((umur_tahun < bawah) | (umur_tahun > atas)).sum())
print(f"     UMUR_TAHUN batas IQR [{bawah:.1f}, {atas:.1f}] -> outlier {n_out} baris")
print(f"     UMUR_TAHUN bernilai 0 tahun: {int((umur_tahun == 0).sum())} baris")
biner = all(set(df[c].unique()) <= {0, 1} for c in list("BCDEFGHIJKLM") + [LABEL])
print(f"     Kolom B-M dan N hanya berisi 0/1: {biner} -> tidak mungkin ada outlier")
print("     -> tidak dibuang: setelah encoding semua umur masuk kode 1-5")

X = df[FITUR].astype(float)
y = df[LABEL]

# 6. Split data (training 80%, testing 20%)
judul("6. SPLIT DATA 80% TRAINING / 20% TESTING")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()                    # standarisasi agar skala fitur seragam
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(f"   Training : {len(X_train)} baris | Testing : {len(X_test)} baris")

# 7. Bangun model klasifikasi
judul("7. BANGUN MODEL")
model = Sequential([
    Dense(16, activation='relu', input_shape=(X.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')  # Output binary
])
model.summary()

# 8. Compile dan train
judul("8. COMPILE DAN TRAIN (100 epoch)")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=1)

plt.figure(figsize=(12, 4))
for i, (m, nama) in enumerate([("accuracy", "Akurasi"), ("loss", "Loss")], 1):
    plt.subplot(1, 2, i)
    plt.plot(history.history[m], label="training")
    plt.plot(history.history["val_" + m], label="validasi")
    plt.xlabel("Epoch")
    plt.ylabel(nama)
    plt.title(f"{nama} per epoch")
    plt.legend()
plt.tight_layout()
simpan_dan_tampilkan("dl_1_history.png")

# 9. Evaluasi
judul("9. EVALUASI")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nAkurasi di data uji: {acc * 100:.2f}%")
print(f"Loss di data uji   : {loss:.4f}")

# 10. Confusion matrix
judul("10. CONFUSION MATRIX")
y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).ravel()   # probabilitas -> kelas 0/1
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 11. Tampilkan confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
simpan_dan_tampilkan("dl_2_confusion_matrix.png")

# 12. Laporan klasifikasi
judul("12. LAPORAN KLASIFIKASI")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
