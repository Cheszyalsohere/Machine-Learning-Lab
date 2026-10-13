# =============================================================
#  UTS PRAKTIKUM MACHINE LEARNING - SOAL E
#  NIM   : 434241079  (Muhammad Irfan Nuha)
#  Kelas : TI-C5  |  Kelompok 5
#  Dataset: Airline Passenger Satisfaction (kategorikal + numerik)
#  Target : satisfaction  (satisfied / neutral or dissatisfied)
#
#  Alur (flowchart Soal E):
#    Input -> Preprocessing (missing, duplikat, outlier)
#    -> Transformasi Min-Max -> Split 80/20
#    -> Seleksi Fitur (2 metode: Chi-Square & ANOVA F)
#    -> Klasifikasi Decision Tree -> Evaluasi (Confusion Matrix,
#       Akurasi, Presisi, Recall, F1-Score)
# =============================================================

import os
import glob
import warnings
warnings.filterwarnings("ignore")     # sembunyikan warning library agar output rapi
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)
TARGET   = "satisfaction"
K_FITUR  = 10            # jumlah fitur teratas yang dipilih tiap metode


def cari_dataset():
    """Cari file dataset .csv di folder (prioritas train.csv)."""
    kandidat = os.path.join(BASE_DIR, "train.csv")
    if os.path.exists(kandidat):
        return kandidat
    csv = [f for f in glob.glob(os.path.join(BASE_DIR, "*.csv"))]
    if not csv:
        raise FileNotFoundError("Tidak ada file .csv di folder. Taruh train.csv di sini.")
    return csv[0]


def garis(t=""):
    print("\n" + "=" * 64)
    if t: print(f"  {t}"); print("=" * 64)


def evaluasi(nama, y_test, y_pred, labels):
    """Cetak confusion matrix + 4 metrik, simpan gambar CM."""
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    print(f"\n  Confusion Matrix ({nama}):")
    print("   ", cm.tolist())
    print(f"  Akurasi : {acc*100:.2f}%")
    print(f"  Presisi : {pre*100:.2f}%")
    print(f"  Recall  : {rec*100:.2f}%")
    print(f"  F1-Score: {f1*100:.2f}%")

    plt.figure(figsize=(4.6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar=False)
    plt.xlabel("Prediksi"); plt.ylabel("Aktual")
    plt.title(f"Confusion Matrix - {nama}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"cm_{nama.lower().replace(' ','_')}.png"), dpi=120)
    plt.close()
    return dict(Metode=nama, Akurasi=acc, Presisi=pre, Recall=rec, F1=f1)


def main():
    # ---------- INPUT ----------
    garis("1. INPUT DATA")
    path = cari_dataset()
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    print(f"File     : {os.path.basename(path)}")
    print(f"Ukuran   : {df.shape[0]} baris x {df.shape[1]} kolom")
    # buang kolom identitas yang tidak berguna
    for c in ["Unnamed: 0", "id", "Id", "ID"]:
        if c in df.columns:
            df = df.drop(columns=c)
    print(f"Kolom    : {list(df.columns)}")
    print("\n5 data teratas:")
    print(df.head().to_string())

    # ---------- PREPROCESSING ----------
    garis("2. PREPROCESSING")

    # (a) Missing value
    miss = df.isna().sum()
    miss = miss[miss > 0]
    print(">> Missing value:")
    if miss.empty:
        print("   Tidak ada missing value.")
    else:
        for k, v in miss.items():
            print(f"   {k}: {v}")

    # (b) Duplikasi
    dup = df.duplicated().sum()
    print(f">> Duplikasi data: {dup} baris")

    # (c) Handling missing + duplikat
    for c in df.select_dtypes(include=[np.number]).columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    for c in df.select_dtypes(include="object").columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode()[0])
    if dup > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    print(f"   -> setelah handling: {df.shape[0]} baris, missing = {int(df.isna().sum().sum())}")

    # (d) Outlier (IQR capping) pada kolom numerik kontinu (nunique > 15)
    kol_num = [c for c in df.select_dtypes(include=[np.number]).columns
               if df[c].nunique() > 15 and c != TARGET]
    Q1 = df[kol_num].quantile(0.25); Q3 = df[kol_num].quantile(0.75)
    IQR = Q3 - Q1
    n_out = int(((df[kol_num] < Q1 - 1.5*IQR) | (df[kol_num] > Q3 + 1.5*IQR)).any(axis=1).sum())
    for c in kol_num:
        df[c] = np.clip(df[c], Q1[c]-1.5*IQR[c], Q3[c]+1.5*IQR[c])
    print(f">> Outlier di-capping (IQR) pada {len(kol_num)} kolom numerik ({n_out} baris terdampak)")

    # (e) Encode kolom kategorikal -> angka
    kol_kat = df.select_dtypes(include="object").columns.tolist()
    encoder = {}
    for c in kol_kat:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c])
        encoder[c] = le
    print(f">> Encoding kategorikal: {kol_kat}")

    # distribusi target
    label_target = list(encoder[TARGET].classes_) if TARGET in encoder else ["0", "1"]
    print(f">> Distribusi target ({TARGET}): {df[TARGET].value_counts().sort_index().to_dict()}")
    print(f"   0 = {label_target[0]}, 1 = {label_target[1]}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # ---------- TRANSFORMASI (MIN-MAX) ----------
    garis("3. TRANSFORMASI (MIN-MAX NORMALIZATION)")
    X_scaled = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns)
    print("   Semua fitur diskalakan ke rentang [0, 1].")
    print(X_scaled.describe().loc[["min", "max"]].round(2).to_string())

    # ---------- SPLIT DATA ----------
    garis("4. SPLIT DATA (80% training / 20% testing)")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y)
    print(f"   Training: {X_train.shape[0]} baris | Testing: {X_test.shape[0]} baris")

    # ---------- SELEKSI FITUR (2 METODE) ----------
    garis(f"5. SELEKSI FITUR (2 metode, ambil {K_FITUR} fitur teratas)")

    hasil_metrik = []
    metode_seleksi = {
        "Chi-Square": chi2,
        "ANOVA F": f_classif,
    }
    for nama, fungsi in metode_seleksi.items():
        selector = SelectKBest(score_func=fungsi, k=min(K_FITUR, X_train.shape[1]))
        selector.fit(X_train, y_train)
        fitur_terpilih = X_train.columns[selector.get_support()].tolist()
        skor = pd.Series(selector.scores_, index=X_train.columns).sort_values(ascending=False)
        print(f"\n>> {nama} - {len(fitur_terpilih)} fitur terpilih:")
        for f in fitur_terpilih:
            print(f"     {f:<32} skor={skor[f]:.2f}")

        # ---------- KLASIFIKASI (DECISION TREE) ----------
        Xtr = selector.transform(X_train)
        Xte = selector.transform(X_test)
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(Xtr, y_train)
        y_pred = dt.predict(Xte)

        # ---------- EVALUASI ----------
        hasil_metrik.append(evaluasi(f"DT + {nama}", y_test, y_pred, label_target))

    # ---------- RINGKASAN ----------
    garis("6. RINGKASAN EVALUASI")
    ringkas = pd.DataFrame(hasil_metrik)
    for c in ["Akurasi", "Presisi", "Recall", "F1"]:
        ringkas[c] = (ringkas[c] * 100).round(2)
    print(ringkas.to_string(index=False))
    ringkas.to_csv(os.path.join(OUT_DIR, "hasil_evaluasi.csv"), index=False)
    print("\nGambar confusion matrix & ringkasan tersimpan di folder output/")


if __name__ == "__main__":
    main()
