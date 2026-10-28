# =============================================================
#  TUGAS KLASIFIKASI - MACHINE LEARNING (Pertemuan 8)
#  File    : klasifikasi_434241079.py
#  Kelompok: 5 (TI-C5) - Muhammad Irfan Nuha (434241079)
#                        Bunga Lala Sal Sabila (434241066)
#  Dataset : Airline Passenger Satisfaction (sama dengan UTS)
#  Target  : satisfaction (neutral or dissatisfied / satisfied)
#
#  Tahapan (sama dengan UTS):
#    Input -> Preprocessing -> Transformasi Min-Max -> Split 80/20
#    -> Seleksi Fitur (Chi-Square & ANOVA F)
#    -> Klasifikasi: Naive Bayes, SVM, Decision Tree
#    -> Evaluasi & perbandingan -> simpan model terbaik (untuk UI)
# =============================================================

import os
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)
TARGET   = "satisfaction"
K_FITUR  = 10


def garis(judul):
    print("\n" + "=" * 64)
    print(f"  {judul}")
    print("=" * 64)


def evaluasi(nama, y_test, y_pred, label):
    """Hitung confusion matrix + metrik, cetak, dan simpan gambar CM."""
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    hasil = {
        "Model": nama,
        "Akurasi": accuracy_score(y_test, y_pred),
        "Presisi": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1": f1_score(y_test, y_pred, average="weighted"),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }
    print(f"     Confusion Matrix : [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]")
    print(f"     Akurasi {hasil['Akurasi']*100:.2f}% | Presisi {hasil['Presisi']*100:.2f}% | "
          f"Recall {hasil['Recall']*100:.2f}% | F1 {hasil['F1']*100:.2f}%")

    plt.figure(figsize=(4.6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=label, yticklabels=label)
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.title(f"Confusion Matrix\n{nama}", fontsize=10)
    plt.tight_layout()
    nama_file = nama.lower().replace(" + ", "_").replace(" ", "_").replace("-", "")
    plt.savefig(os.path.join(OUT_DIR, f"cm_{nama_file}.png"), dpi=120)
    plt.close()
    return hasil


def main():
    # ---------------- 1. INPUT ----------------
    garis("1. INPUT DATA")
    df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
    df.columns = df.columns.str.strip()
    df = df.drop(columns=[c for c in ["Unnamed: 0", "id"] if c in df.columns])
    print(f"Ukuran data : {df.shape[0]} baris x {df.shape[1]} kolom")
    print(df.iloc[:5, :6].to_string(index=False))
    print(f"... (+{df.shape[1] - 6} kolom lainnya)")

    # ---------------- 2. PREPROCESSING ----------------
    garis("2. PREPROCESSING")
    miss = df.isna().sum()
    print(f">> Missing value : {int(miss.sum())} sel "
          f"({', '.join(f'{k}={v}' for k, v in miss[miss > 0].items()) or '-'})")
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].fillna(df[c].median())
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].fillna(df[c].mode()[0])

    dup = int(df.duplicated().sum())
    print(f">> Duplikasi     : {dup} baris")
    df = df.drop_duplicates().reset_index(drop=True)

    kol_num = [c for c in df.select_dtypes(include=[np.number]).columns
               if df[c].nunique() > 15]
    Q1, Q3 = df[kol_num].quantile(0.25), df[kol_num].quantile(0.75)
    IQR = Q3 - Q1
    batas = {c: (Q1[c] - 1.5 * IQR[c], Q3[c] + 1.5 * IQR[c]) for c in kol_num}
    for c in kol_num:
        df[c] = np.clip(df[c], *batas[c])
    print(f">> Outlier       : di-capping IQR pada {len(kol_num)} kolom numerik")

    encoder = {}
    for c in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c])
        encoder[c] = list(le.classes_)
    label = encoder[TARGET]
    print(f">> Encoding      : {[c for c in encoder if c != TARGET]}")
    print(f">> Target        : 0 = {label[0]} ({(df[TARGET] == 0).sum()}), "
          f"1 = {label[1]} ({(df[TARGET] == 1).sum()})")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # ---------------- 3. TRANSFORMASI ----------------
    garis("3. TRANSFORMASI MIN-MAX")
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    print(f"   {X.shape[1]} fitur diskalakan ke rentang "
          f"[{X_scaled.values.min():.0f}, {X_scaled.values.max():.0f}]")

    # ---------------- 4. SPLIT ----------------
    garis("4. SPLIT DATA 80% TRAINING / 20% TESTING")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y)
    print(f"   Training : {len(X_train)} baris")
    print(f"   Testing  : {len(X_test)} baris")

    # ---------------- 5-7. SELEKSI FITUR + KLASIFIKASI + EVALUASI ----------------
    semua_hasil, model_terlatih = [], {}
    for nama_seleksi, fungsi in [("Chi-Square", chi2), ("ANOVA F", f_classif)]:
        garis(f"5. SELEKSI FITUR: {nama_seleksi} ({K_FITUR} fitur teratas)")
        selector = SelectKBest(score_func=fungsi, k=K_FITUR).fit(X_train, y_train)
        fitur = X_train.columns[selector.get_support()].tolist()
        for f in fitur:
            print(f"   - {f}")
        Xtr, Xte = selector.transform(X_train), selector.transform(X_test)

        garis(f"6. KLASIFIKASI & EVALUASI (fitur {nama_seleksi})")
        daftar_model = {
            "Naive Bayes": GaussianNB(),
            "SVM": SVC(kernel="rbf", random_state=42, cache_size=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
        }
        for nama_model, model in daftar_model.items():
            if nama_model == "SVM":
                print(f">> {nama_model} sedang dilatih (bisa 2-3 menit)...")
            mulai = time.time()
            model.fit(Xtr, y_train)
            durasi = time.time() - mulai
            nama = f"{nama_model} + {nama_seleksi}"
            print(f">> {nama}  (waktu latih {durasi:.1f} detik)")
            hasil = evaluasi(nama, y_test, model.predict(Xte), label)
            hasil["Waktu (detik)"] = round(durasi, 1)
            semua_hasil.append(hasil)
            model_terlatih[nama] = (model, selector, fitur)

    # ---------------- 8. PERBANDINGAN ----------------
    garis("7. PERBANDINGAN KINERJA (urut akurasi)")
    tabel = pd.DataFrame(semua_hasil).sort_values("Akurasi", ascending=False)
    tampil = tabel[["Model", "Akurasi", "Presisi", "Recall", "F1"]].copy()
    for c in ["Akurasi", "Presisi", "Recall", "F1"]:
        tampil[c] = (tampil[c] * 100).round(2)
    print(tampil.to_string(index=False))
    tabel.to_csv(os.path.join(OUT_DIR, "perbandingan_model.csv"), index=False)

    plt.figure(figsize=(9, 4.5))
    warna = {"Naive Bayes": "#8fb3e0", "SVM": "#3d7cc9", "Decision Tree": "#f0a35e"}
    urut = tabel.sort_values("Model")
    bar = plt.barh(urut["Model"], urut["Akurasi"] * 100,
                   color=[warna[m.split(" + ")[0]] for m in urut["Model"]])
    plt.bar_label(bar, fmt="%.2f%%", padding=3, fontsize=8)
    plt.xlim(80, 96)
    plt.xlabel("Akurasi (%)")
    plt.title("Perbandingan Akurasi Naive Bayes, SVM, dan Decision Tree")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "perbandingan_akurasi.png"), dpi=120)
    plt.close()

    # ---------------- 9. SIMPAN MODEL TERBAIK ----------------
    terbaik = tabel.iloc[0]["Model"]
    model, selector, fitur = model_terlatih[terbaik]
    bundle = {
        "nama_model": terbaik,
        "akurasi": float(tabel.iloc[0]["Akurasi"]),
        "model": model,
        "scaler": scaler,
        "selector": selector,
        "urutan_fitur": list(X.columns),
        "fitur_terpilih": fitur,
        "encoder": {k: v for k, v in encoder.items() if k != TARGET},
        "label": label,
        "default": X.median().to_dict(),
        "batas_outlier": batas,
        "rentang": {c: (float(X[c].min()), float(X[c].max())) for c in X.columns},
        "perbandingan": tampil.to_dict("records"),
    }
    joblib.dump(bundle, os.path.join(BASE_DIR, "model_terbaik.joblib"))
    garis("8. MODEL TERBAIK")
    print(f"   {terbaik}  ->  akurasi {tabel.iloc[0]['Akurasi'] * 100:.2f}%")
    print("   Disimpan ke model_terbaik.joblib (dipakai oleh UI Streamlit)")


if __name__ == "__main__":
    main()
