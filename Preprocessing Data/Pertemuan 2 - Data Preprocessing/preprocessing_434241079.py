# =============================================================
#  TUGAS DATA PREPROCESSING - MACHINE LEARNING
#  Nama File : preprocessing_434241079.py
#  NIM       : 434241079
#  Materi    : Pertemuan 2 - Data Preprocessing
#  Deskripsi : Read Data & Data Preprocessing untuk 3 dataset
#              sekaligus (CC GENERAL, concrete_data, heart):
#                1. Cek Missing Value
#                2. Handling Missing Value (median numerik, mode kategori)
#                3. Deteksi Outlier (Z-Score & Boxplot/IQR)
#                4. Handling Outlier (Capping batas IQR)
# =============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # backend non-GUI untuk simpan gambar
import matplotlib.pyplot as plt
from scipy import stats

# --- Opsi tampilan pandas supaya cetakan tidak berantakan ---
pd.set_option("display.width", 100)
pd.set_option("display.max_columns", 8)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = ["CC GENERAL.csv", "concrete_data.csv", "heart.csv"]


# ==================================================================
#  UTILITAS TAMPILAN
# ==================================================================
def judul(teks):
    """Header dataset dalam bingkai agar mudah dibedakan."""
    print("\n" + "=" * 60)
    print(f"  {teks}")
    print("=" * 60)


def sub(nomor, teks):
    """Sub-judul tiap tahap preprocessing."""
    print(f"\n[{nomor}] {teks}")


def preview_data(df, n=5, maks_kolom=6):
    """Menampilkan cuplikan data secara ringkas (tidak melebar)."""
    tampil = df.iloc[:n, :maks_kolom]
    print(f"    Cuplikan {n} baris pertama ({maks_kolom} kolom awal):")
    for baris in tampil.to_string(index=False).splitlines():
        print("      " + baris)
    sisa = df.shape[1] - maks_kolom
    if sisa > 0:
        print(f"      ... (+{sisa} kolom lainnya)")


def kolom_kontinu(df):
    """Kolom numerik kontinu (nunique > 15) untuk analisis outlier."""
    num = df.select_dtypes(include=[np.number])
    return [c for c in num.columns if df[c].nunique() > 15]


# ==================================================================
#  TAHAP-TAHAP PREPROCESSING
# ==================================================================
def cek_missing_value(df):
    sub(1, "CEK MISSING VALUE")
    miss = df.isnull().sum()
    kosong = miss[miss > 0].sort_values(ascending=False)
    if kosong.empty:
        print("    Tidak ada nilai kosong pada seluruh kolom.")
    else:
        lebar = max(len(k) for k in kosong.index)
        print("    Kolom yang memiliki nilai kosong:")
        for kol, jml in kosong.items():
            print(f"      {kol:<{lebar}} : {jml}")
    print(f"    Total sel kosong : {int(miss.sum())}")


def handling_missing_value(df):
    sub(2, "HANDLING MISSING VALUE")
    df = df.copy()
    dilakukan = False

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)
            print(f"      {col:<20} -> diisi median = {median:,.3f}")
            dilakukan = True

    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isnull().any():
            mode = df[col].mode()[0]
            df[col] = df[col].fillna(mode)
            print(f"      {col:<20} -> diisi mode   = {mode}")
            dilakukan = True

    if not dilakukan:
        print("      (tidak ada yang perlu diisi)")
    print(f"    Sisa missing value : {int(df.isnull().sum().sum())}")
    return df


def deteksi_outlier(df, kolom, nama):
    sub(3, f"DETEKSI OUTLIER ({len(kolom)} kolom numerik kontinu)")

    # (a) Z-Score, threshold |z| > 3
    z = np.abs(stats.zscore(df[kolom]))
    n_z = int((z > 3).any(axis=1).sum())

    # (b) IQR + simpan boxplot "sebelum"
    Q1 = df[kolom].quantile(0.25)
    Q3 = df[kolom].quantile(0.75)
    IQR = Q3 - Q1
    batas_bawah = Q1 - 1.5 * IQR
    batas_atas  = Q3 + 1.5 * IQR
    n_iqr = int(((df[kolom] < batas_bawah) | (df[kolom] > batas_atas)).any(axis=1).sum())

    print(f"      Z-Score (|z| > 3) : {n_z} baris outlier")
    print(f"      IQR (1.5 x IQR)   : {n_iqr} baris outlier")
    simpan_boxplot(df, kolom, nama, "sebelum")
    return batas_bawah, batas_atas


def handling_outlier(df, kolom, batas_bawah, batas_atas, nama):
    sub(4, "HANDLING OUTLIER (capping ke batas IQR)")
    df = df.copy()
    for col in kolom:
        df[col] = np.where(df[col] < batas_bawah[col], batas_bawah[col], df[col])
        df[col] = np.where(df[col] > batas_atas[col],  batas_atas[col],  df[col])
    print("      Nilai ekstrem diganti dengan batas bawah/atas IQR.")
    simpan_boxplot(df, kolom, nama, "sesudah")
    return df


def simpan_boxplot(df, kolom, nama, tahap):
    plt.figure(figsize=(max(6, len(kolom) * 0.9), 4.5))
    df[kolom].boxplot(rot=45, grid=False)
    plt.title(f"Boxplot {tahap} handling outlier - {nama}")
    plt.tight_layout()
    fn = os.path.join(OUT_DIR, f"boxplot_{tahap}_{nama.replace('.csv','').replace(' ','_')}.png")
    plt.savefig(fn, dpi=110)
    plt.close()


# ==================================================================
#  PIPELINE SATU DATASET
# ==================================================================
def proses_dataset(nama_file):
    judul(f"DATASET: {nama_file}")

    df = pd.read_csv(os.path.join(BASE_DIR, nama_file))
    print(f"Ukuran data : {df.shape[0]} baris x {df.shape[1]} kolom\n")
    preview_data(df)

    cek_missing_value(df)
    df = handling_missing_value(df)

    kolom = kolom_kontinu(df)
    bb, ba = deteksi_outlier(df, kolom, nama_file)
    df = handling_outlier(df, kolom, bb, ba, nama_file)

    out_csv = os.path.join(OUT_DIR, f"clean_{nama_file}")
    df.to_csv(out_csv, index=False)
    print(f"\n    Hasil    -> output/clean_{nama_file}")
    print(f"    Boxplot  -> output/boxplot_sebelum & sesudah_{nama_file.replace('.csv','')}.png")
    return df


# ==================================================================
#  MAIN
# ==================================================================
if __name__ == "__main__":
    print("PROGRAM DATA PREPROCESSING  |  NIM 434241079")
    print(f"Memproses {len(DATASETS)} dataset sekaligus ...")
    for ds in DATASETS:
        proses_dataset(ds)
    print("\n" + "=" * 60)
    print("  SELESAI - Semua dataset telah di-preprocessing.")
    print("=" * 60)
