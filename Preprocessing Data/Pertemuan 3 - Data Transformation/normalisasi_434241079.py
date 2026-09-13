# =============================================================
#  TUGAS TRANSFORMASI / NORMALISASI DATA - MACHINE LEARNING
#  Nama File : normalisasi_434241079.py
#  NIM       : 434241079
#  Materi    : Pertemuan 3 - Data Transformation
#  Alur      : (1) Preprocessing dulu (handling missing + outlier)
#              (2) Normalisasi 3 metode:
#                   - Simple Feature Scaling : x' = x / max(x)
#                   - Min-Max Normalization  : x' = (x - min)/(max - min)
#                   - Z-Score Standardization: x' = (x - mean)/std
#              Diterapkan pada 3 dataset: CC GENERAL, concrete, heart.
# =============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

pd.set_option("display.width", 100)
pd.set_option("display.max_columns", 8)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = ["CC GENERAL.csv", "concrete_data.csv", "heart.csv"]


# ==================================================================
#  UTILITAS TAMPILAN
# ==================================================================
def judul(teks):
    print("\n" + "=" * 62)
    print(f"  {teks}")
    print("=" * 62)


def sub(teks):
    print(f"\n-- {teks}")


def kolom_kontinu(df):
    """Kolom numerik kontinu (nunique > 15) untuk dinormalisasi."""
    num = df.select_dtypes(include=[np.number])
    return [c for c in num.columns if df[c].nunique() > 15]


# ==================================================================
#  TAHAP 1 - PREPROCESSING (ringkas, dari materi Pertemuan 2)
# ==================================================================
def preprocessing(df):
    # a. Handling missing value: numerik -> median, kategori -> mode
    n_missing = int(df.isnull().sum().sum())
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # b. Handling outlier: capping ke batas IQR pada kolom kontinu
    kol = kolom_kontinu(df)
    Q1 = df[kol].quantile(0.25)
    Q3 = df[kol].quantile(0.75)
    IQR = Q3 - Q1
    bb, ba = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    for c in kol:
        df[c] = np.clip(df[c], bb[c], ba[c])

    print(f"   Missing value ditangani : {n_missing} sel -> 0")
    print(f"   Outlier di-capping pada  : {len(kol)} kolom kontinu")
    return df, kol


# ==================================================================
#  TAHAP 2 - NORMALISASI (3 metode)
# ==================================================================
def simple_feature_scaling(s):
    return s / s.max()

def min_max_normalization(s):
    return (s - s.min()) / (s.max() - s.min())

def z_score_standardization(s):
    return (s - s.mean()) / s.std(ddof=0)     # ddof=0 = std populasi (sesuai scipy)


def ringkas_rentang(df, kolom, label):
    """Menampilkan min | max | mean | std tiap kolom (bukti hasil normalisasi)."""
    def z0(x):                       # hindari tampilan "-0.000"
        return 0.0 if abs(x) < 0.0005 else x
    print(f"   {label}")
    print(f"      {'kolom':<24}{'min':>10}{'max':>10}{'mean':>10}{'std':>10}")
    for c in kolom:
        s = df[c]
        print(f"      {c[:23]:<24}{z0(s.min()):>10.3f}{z0(s.max()):>10.3f}"
              f"{z0(s.mean()):>10.3f}{z0(s.std(ddof=0)):>10.3f}")


def plot_perbandingan(asli, sfs, mm, zsc, kolom, nama):
    """Histogram 1 kolom perwakilan: sebelum vs 3 metode normalisasi."""
    c = kolom[0]
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.2))
    for a, data, judulp in zip(
        ax, [asli[c], sfs[c], mm[c], zsc[c]],
        [f"Asli\n({c[:18]})", "Simple Feature\nScaling", "Min-Max", "Z-Score"]):
        a.hist(data, bins=30, color="#3b6ea5", edgecolor="white")
        a.set_title(judulp, fontsize=9)
        a.tick_params(labelsize=7)
    fig.suptitle(f"Distribusi kolom '{c}' sebelum & sesudah normalisasi - {nama}", fontsize=10)
    plt.tight_layout()
    fn = os.path.join(OUT_DIR, f"normalisasi_{nama.replace('.csv','').replace(' ','_')}.png")
    plt.savefig(fn, dpi=110)
    plt.close()


# ==================================================================
#  PIPELINE SATU DATASET
# ==================================================================
def proses(nama_file):
    judul(f"DATASET: {nama_file}")
    df = pd.read_csv(os.path.join(BASE_DIR, nama_file))
    print(f"Ukuran data : {df.shape[0]} baris x {df.shape[1]} kolom")

    sub("TAHAP 1: PREPROCESSING")
    df, kolom = preprocessing(df)

    sub("TAHAP 2: NORMALISASI (kolom kontinu)")
    ringkas_rentang(df, kolom, "Rentang data ASLI (sebelum normalisasi):")

    sfs = df.copy(); mm = df.copy(); zsc = df.copy()
    for c in kolom:
        sfs[c] = simple_feature_scaling(df[c])
        mm[c]  = min_max_normalization(df[c])
        zsc[c] = z_score_standardization(df[c])

    print()
    ringkas_rentang(sfs, kolom, "1) Simple Feature Scaling  (harusnya max = 1):")
    print()
    ringkas_rentang(mm, kolom, "2) Min-Max Normalization   (harusnya min=0, max=1):")
    print()
    ringkas_rentang(zsc, kolom, "3) Z-Score Standardization (harusnya mean=0, std=1):")

    # simpan hasil (contoh: Min-Max) + gambar perbandingan
    key = nama_file.replace(".csv", "")
    mm.to_csv(os.path.join(OUT_DIR, f"normalisasi_minmax_{nama_file}"), index=False)
    zsc.to_csv(os.path.join(OUT_DIR, f"normalisasi_zscore_{nama_file}"), index=False)
    sfs.to_csv(os.path.join(OUT_DIR, f"normalisasi_sfs_{nama_file}"), index=False)
    plot_perbandingan(df, sfs, mm, zsc, kolom, nama_file)
    print(f"\n   Hasil disimpan -> output/normalisasi_[sfs|minmax|zscore]_{nama_file}")
    print(f"   Grafik         -> output/normalisasi_{key}.png")


# ==================================================================
#  CONTOH PERHITUNGAN MANUAL (sesuai contoh di modul: X=[7,10,15,20,25])
# ==================================================================
def contoh_modul():
    judul("CONTOH PERHITUNGAN (X = [7, 10, 15, 20, 25])")
    X = pd.Series([7, 10, 15, 20, 25], dtype=float)
    tabel = pd.DataFrame({
        "X": X,
        "Simple Feature Scaling": simple_feature_scaling(X).round(4),
        "Min-Max": min_max_normalization(X).round(4),
        "Z-Score": z_score_standardization(X).round(4),
    })
    print(tabel.to_string(index=False))


# ==================================================================
#  MAIN
# ==================================================================
if __name__ == "__main__":
    print("PROGRAM NORMALISASI DATA  |  NIM 434241079")
    print(f"Preprocessing + Normalisasi 3 metode pada {len(DATASETS)} dataset ...")
    contoh_modul()
    for ds in DATASETS:
        proses(ds)
    print("\n" + "=" * 62)
    print("  SELESAI - Semua dataset telah dinormalisasi.")
    print("=" * 62)
