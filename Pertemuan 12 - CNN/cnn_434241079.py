# =============================================================
#  TUGAS CNN - MACHINE LEARNING (Pertemuan 12)
#  File    : cnn_434241079.py
#  Nama    : Muhammad Irfan Nuha  |  NIM : 434241079  |  TI-C5
#  Dataset : EuroSAT RGB - citra satelit Sentinel-2, 27.000 gambar
#            64x64 piksel, 10 kelas tata guna lahan
#            https://zenodo.org/records/7711810
#  Model   : Convolutional Neural Network (Keras Sequential)
#            4 blok konvolusi + Global Average Pooling
# =============================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"        # sembunyikan log internal TensorFlow
import time
import warnings
warnings.filterwarnings("ignore")

# 1. Import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

tf.keras.utils.set_random_seed(42)               # agar hasil training bisa diulang

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "EuroSAT_RGB")
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

ARTI = {"AnnualCrop": "tanaman semusim", "Forest": "hutan", "HerbaceousVegetation": "vegetasi herba",
        "Highway": "jalan raya", "Industrial": "kawasan industri", "Pasture": "padang rumput",
        "PermanentCrop": "tanaman tahunan", "Residential": "perumahan", "River": "sungai",
        "SeaLake": "laut / danau"}


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
KELAS = sorted(os.listdir(DATA_DIR))
paths, labels = [], []
for i, k in enumerate(KELAS):
    for f in sorted(os.listdir(os.path.join(DATA_DIR, k))):
        paths.append(os.path.join(DATA_DIR, k, f))
        labels.append(i)
labels = np.array(labels)
print(f"Jumlah gambar : {len(paths)} | Jumlah kelas : {len(KELAS)}")
for i, k in enumerate(KELAS):
    print(f"   {i}. {k:<21} ({ARTI[k]:<16}): {(labels == i).sum()} gambar")

# 3. Baca gambar menjadi array
judul("3. BACA GAMBAR")
mulai = time.time()
X, ukuran, rusak = [], set(), 0
for p in paths:
    try:
        img = Image.open(p).convert("RGB")
    except OSError:
        rusak += 1
        continue
    ukuran.add(img.size)
    X.append(np.asarray(img, dtype=np.uint8))
X = np.stack(X)
print(f"   Shape data     : {X.shape}  (gambar, tinggi, lebar, channel RGB)")
print(f"   Ukuran gambar  : {ukuran}")
print(f"   File rusak     : {rusak}")
print(f"   Rentang piksel : {X.min()} - {X.max()} (dinormalisasi ke 0-1 di dalam model)")
print(f"   Waktu baca     : {time.time() - mulai:.1f} detik")

plt.figure(figsize=(12, 5))
for i, k in enumerate(KELAS):
    for j in range(2):
        plt.subplot(2, 10, j * 10 + i + 1)
        plt.imshow(X[np.where(labels == i)[0][j * 7]])
        plt.axis("off")
        if j == 0:
            plt.title(k, fontsize=8)
plt.suptitle("Contoh gambar tiap kelas EuroSAT")
plt.tight_layout()
simpan_dan_tampilkan("cnn_1_contoh_gambar.png")

# 4. Split data (training 70%, validasi 15%, testing 15%)
judul("4. SPLIT DATA 70% TRAIN / 15% VALIDASI / 15% TEST")
X_train, X_sisa, y_train, y_sisa = train_test_split(X, labels, test_size=0.30, random_state=42, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(X_sisa, y_sisa, test_size=0.50, random_state=42, stratify=y_sisa)
print(f"   Training : {len(X_train)} gambar")
print(f"   Validasi : {len(X_val)} gambar")
print(f"   Testing  : {len(X_test)} gambar")

# 5. Bangun arsitektur CNN
judul("5. ARSITEKTUR CNN")
augmentasi = models.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),        # citra satelit tidak punya arah atas/bawah
    layers.RandomRotation(0.1, fill_mode="reflect"),
], name="augmentasi")


def blok_konvolusi(filter_):
    return [layers.Conv2D(filter_, 3, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu")]


model = models.Sequential([
    layers.Input(shape=(64, 64, 3)),
    augmentasi,
    layers.Rescaling(1.0 / 255),                         # piksel 0-255 -> 0-1
    *blok_konvolusi(32), layers.MaxPooling2D(),          # 64x64 -> 32x32
    *blok_konvolusi(64), layers.MaxPooling2D(),          # 32x32 -> 16x16
    *blok_konvolusi(128), layers.MaxPooling2D(),         # 16x16 -> 8x8
    layers.Dropout(0.25),
    *blok_konvolusi(256),
    layers.GlobalAveragePooling2D(),                     # 8x8x256 -> 256
    layers.Dropout(0.4),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(KELAS), activation="softmax"),      # probabilitas 10 kelas
], name="cnn_eurosat")
model.summary()

# 6. Compile dan train
judul("6. COMPILE DAN TRAIN")
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
henti = callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
turunkan_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
mulai = time.time()
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=40, batch_size=64,
                    callbacks=[henti, turunkan_lr], verbose=2)
print(f"\nTraining selesai: {len(history.history['loss'])} epoch, {time.time() - mulai:.0f} detik")
print(f"Bobot terbaik dari epoch {int(np.argmin(history.history['val_loss'])) + 1} dipakai (EarlyStopping)")

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
simpan_dan_tampilkan("cnn_2_history.png")

# 7. Evaluasi
judul("7. EVALUASI DATA TEST")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Akurasi di data uji: {acc * 100:.2f}%")
print(f"Loss di data uji   : {loss:.4f}")

# 8. Confusion matrix
judul("8. CONFUSION MATRIX")
y_prob = model.predict(X_test, verbose=0)
y_pred = y_prob.argmax(axis=1)                          # kelas dengan probabilitas terbesar
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm, index=[k[:6] for k in KELAS], columns=[k[:6] for k in KELAS]).to_string())

plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=KELAS, yticklabels=KELAS)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix CNN EuroSAT")
simpan_dan_tampilkan("cnn_3_confusion_matrix.png")

salah = [(KELAS[a], KELAS[b], cm[a, b]) for a in range(len(KELAS)) for b in range(len(KELAS)) if a != b]
print("\nPasangan kelas yang paling sering tertukar (aktual -> prediksi):")
for a, b, n in sorted(salah, key=lambda t: -t[2])[:5]:
    print(f"   {a:<21} -> {b:<21}: {n} gambar")

# 9. Laporan klasifikasi
judul("9. LAPORAN KLASIFIKASI")
print(classification_report(y_test, y_pred, target_names=KELAS, digits=4))

# 10. Contoh prediksi
judul("10. CONTOH PREDIKSI")
rng = np.random.default_rng(7)
contoh = np.concatenate([rng.choice(np.where(y_pred == y_test)[0], 9, replace=False),
                         rng.choice(np.where(y_pred != y_test)[0], 3, replace=False)])
plt.figure(figsize=(12, 7))
for n, i in enumerate(contoh, 1):
    plt.subplot(3, 4, n)
    plt.imshow(X_test[i])
    plt.axis("off")
    benar = y_pred[i] == y_test[i]
    plt.title(f"Asli: {KELAS[y_test[i]]}\nPrediksi: {KELAS[y_pred[i]]} ({y_prob[i].max() * 100:.0f}%)",
              fontsize=8, color="green" if benar else "red")
    print(f"   {'BENAR' if benar else 'SALAH'} | asli {KELAS[y_test[i]]:<21} | prediksi "
          f"{KELAS[y_pred[i]]:<21} | yakin {y_prob[i].max() * 100:5.1f}%")
plt.suptitle("Contoh prediksi data uji (hijau = benar, merah = salah)")
plt.tight_layout()
simpan_dan_tampilkan("cnn_4_contoh_prediksi.png")

# 11. Simpan model
model.save(os.path.join(OUT_DIR, "model_cnn_eurosat.keras"))
print("\nModel disimpan -> output/model_cnn_eurosat.keras")
