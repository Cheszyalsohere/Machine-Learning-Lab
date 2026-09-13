"""
=============================================================================
TUGAS DATA PREPROCESSING  -  Pertemuan 2
=============================================================================
Nama  : Muhammad Irfan Nuha
NIM   : 434241079
Kelas : TI-C5

Dataset yang digunakan (3 buah):
  1. Concrete Compressive Strength  (concrete_data.csv)   - 1030 baris, 9 kolom
  2. Credit Card Customer Segment   (CC GENERAL.csv)      - 8950 baris, 18 kolom
  3. Heart Failure Prediction       (heart.csv)           -  918 baris, 12 kolom

Tahapan yang dikerjakan (sesuai modul):
  A. Read Data
  B. Cek Missing Value
  C. Handling Missing Value
  D. Deteksi Outlier (Z-Score & IQR/Boxplot)
  E. Handling Outlier
=============================================================================
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")          # supaya bisa simpan gambar tanpa display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
sns.set_theme(style="whitegrid")

DATA_DIR = "data"
FIG_DIR = "figures"
OUT_DIR = "output"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# FUNGSI BANTU (dipakai untuk ketiga dataset)
# =============================================================================
def judul(teks, char="="):
    print("\n" + char * 78)
    print(teks)
    print(char * 78)


def cek_missing(df, nama=""):
    """B. Cek missing value -> jumlah dan persentase NaN per kolom."""
    n = df.isnull().sum()
    tabel = pd.DataFrame({"jumlah_missing": n,
                          "persen": (n / len(df) * 100).round(2)})
    tabel = tabel[tabel.jumlah_missing > 0]
    print(f"[Cek Missing Value] {nama}")
    if tabel.empty:
        print("  Tidak ada NaN eksplisit terdeteksi (isnull() == 0 untuk semua kolom).")
    else:
        print(tabel.to_string())
    return tabel


def zscore_manual(X):
    """
    Z-score dihitung manual dengan pandas supaya:
      - hasilnya tetap DataFrame (bisa diindeks pakai NAMA KOLOM)
      - NaN diabaikan, tidak membuat seluruh kolom jadi NaN
    scipy.stats.zscore mengembalikan numpy array + gagal kalau ada NaN.
    """
    return (X - X.mean()) / X.std(ddof=0)


def batas_iqr(X):
    """Hitung batas bawah & batas atas metode IQR (Tukey fence 1.5*IQR)."""
    q1 = X.quantile(0.25)
    q3 = X.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def deteksi_outlier(df, kolom, nama=""):
    """D. Deteksi outlier: bandingkan metode Z-Score (>3) dan IQR."""
    X = df[kolom]
    z = zscore_manual(X).abs()
    lo, hi = batas_iqr(X)

    mask_z = z > 3
    mask_iqr = (X < lo) | (X > hi)

    ringkas = pd.DataFrame({
        "skewness": X.skew().round(2),
        "outlier_Zscore": mask_z.sum(),
        "%_Z": (mask_z.sum() / len(X) * 100).round(1),
        "outlier_IQR": mask_iqr.sum(),
        "%_IQR": (mask_iqr.sum() / len(X) * 100).round(1),
        "batas_bawah_IQR": lo.round(2),
        "batas_atas_IQR": hi.round(2),
    })
    print(f"[Deteksi Outlier] {nama}")
    print(ringkas.to_string())
    print(f"\n  Baris terbuang jika DROP semua Z>3  : {mask_z.any(axis=1).sum():>5} "
          f"({mask_z.any(axis=1).mean() * 100:.1f}% dari data)")
    print(f"  Baris terbuang jika DROP semua IQR  : {mask_iqr.any(axis=1).sum():>5} "
          f"({mask_iqr.any(axis=1).mean() * 100:.1f}% dari data)")
    return ringkas, mask_z, mask_iqr


def capping_iqr(df, kolom):
    """
    E. Handling outlier dengan CAPPING (winsorizing):
    nilai di luar pagar IQR digeser ke batas, baris TIDAK dibuang.
    """
    out = df.copy()
    lo, hi = batas_iqr(out[kolom])
    for col in kolom:
        out[col] = np.where(out[col] < lo[col], lo[col], out[col])
        out[col] = np.where(out[col] > hi[col], hi[col], out[col])
    return out


def boxplot_before_after(df_before, df_after, kolom, judul_gbr, fname, log=False):
    """Boxplot sebelum vs sesudah handling outlier -> disimpan sebagai PNG."""
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True)
    for a, d, t in zip(ax, [df_before, df_after], ["SEBELUM", "SESUDAH"]):
        sns.boxplot(data=d[kolom], orient="h", ax=a, fliersize=2)
        a.set_title(f"{t} handling outlier", fontsize=11, weight="bold")
        if log:
            a.set_xscale("symlog")
            a.set_xlabel("nilai (skala symlog)")
    fig.suptitle(judul_gbr, fontsize=13, weight="bold")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> boxplot disimpan: {path}")


ringkasan_akhir = []


# =============================================================================
# DATASET 1 : CONCRETE COMPRESSIVE STRENGTH
# =============================================================================
judul("DATASET 1 : CONCRETE COMPRESSIVE STRENGTH")

# ---- A. Read Data --------------------------------------------------------
df1 = pd.read_csv(os.path.join(DATA_DIR, "concrete_data.csv"))
print("Shape awal :", df1.shape)

# CATATAN: nama kolom di file asli mengandung spasi ('fine_aggregate ').
# Kalau tidak dibersihkan, df['fine_aggregate'] akan error KeyError.
print("Nama kolom sebelum dibersihkan :", list(df1.columns)[6:8])
df1.columns = df1.columns.str.strip()
print("Nama kolom setelah dibersihkan :", list(df1.columns)[6:8])
print("\n5 baris pertama:")
print(df1.head().to_string())
print("\nInfo tipe data:")
print(df1.dtypes.to_string())

# ---- B. Cek Missing Value -----------------------------------------------
judul("1B. CEK MISSING VALUE", "-")
cek_missing(df1, "Concrete")

# Cek "missing terselubung": nilai 0 pada kolom numerik
print("\n[Cek nilai 0 per kolom]")
print((df1 == 0).sum().to_string())
print("""
INTERPRETASI: nilai 0 pada blast_furnace_slag / fly_ash / superplasticizer
BUKAN missing value. Itu artinya campuran beton tersebut memang tidak memakai
bahan tersebut (nol adalah nilai yang sah). Jadi TIDAK boleh diimputasi.""")

# ---- C. Handling Missing Value ------------------------------------------
judul("1C. HANDLING MISSING VALUE + DUPLIKAT", "-")
print("Tidak ada missing value -> tidak ada imputasi yang dilakukan.")
dup = df1.duplicated().sum()
print(f"\nJumlah baris duplikat penuh : {dup}")
print("Contoh baris duplikat:")
print(df1[df1.duplicated(keep=False)].sort_values(list(df1.columns)).head(4).to_string())
df1_clean = df1.drop_duplicates().reset_index(drop=True)
print(f"\nShape setelah drop duplikat : {df1_clean.shape}  (dari {df1.shape})")

# ---- D. Deteksi Outlier --------------------------------------------------
judul("1D. DETEKSI OUTLIER (Z-Score & IQR)", "-")
kol1 = list(df1_clean.columns)
_, mz1, mi1 = deteksi_outlier(df1_clean, kol1, "Concrete")

print("\n[Nilai unik kolom 'age' - jumlah hari curing]")
print(df1_clean["age"].value_counts().sort_index().to_string())
print("""
INTERPRETASI PENTING: 'age' hanya berisi 14 nilai diskrit (1, 3, 7, 14, 28, 56,
90, 91, 100, 120, 180, 270, 360, 365 hari). Ini adalah TITIK DESAIN EKSPERIMEN,
bukan kesalahan pengukuran. Membuang 'outlier' age = membuang seluruh percobaan
curing jangka panjang. Jadi kolom age SENGAJA TIDAK di-treatment.""")

# ---- E. Handling Outlier -------------------------------------------------
judul("1E. HANDLING OUTLIER", "-")
kol1_cap = [c for c in kol1 if c not in ("age", "concrete_compressive_strength")]
print("Strategi : capping IQR pada variabel komposisi bahan saja.")
print("Dikecualikan : 'age' (variabel desain) dan target 'concrete_compressive_strength'.")
df1_final = capping_iqr(df1_clean, kol1_cap)
n_ubah1 = (df1_final[kol1_cap] != df1_clean[kol1_cap]).sum().sum()
print(f"Jumlah nilai yang digeser ke batas : {n_ubah1}")
boxplot_before_after(df1_clean, df1_final, kol1_cap,
                     "Dataset 1 - Concrete Compressive Strength",
                     "01_concrete_boxplot.png")
df1_final.to_csv(os.path.join(OUT_DIR, "concrete_clean.csv"), index=False)
ringkasan_akhir.append(["Concrete", df1.shape, df1_final.shape, 0, dup, n_ubah1])


# =============================================================================
# DATASET 2 : CREDIT CARD CUSTOMER
# =============================================================================
judul("DATASET 2 : CREDIT CARD CUSTOMER")

# ---- A. Read Data --------------------------------------------------------
df2 = pd.read_csv(os.path.join(DATA_DIR, "CC GENERAL.csv"))
print("Shape awal :", df2.shape)
print("\n5 baris pertama:")
print(df2.head().to_string())
print("\nInfo tipe data:")
print(df2.dtypes.to_string())

# CUST_ID adalah identitas, bukan variabel numerik -> harus dikeluarkan
kol2 = [c for c in df2.columns if c != "CUST_ID"]
print("\nCUST_ID dikeluarkan dari analisis numerik (kolom identitas, bukan fitur).")

# ---- B. Cek Missing Value -----------------------------------------------
judul("2B. CEK MISSING VALUE", "-")
cek_missing(df2, "Credit Card")

# ---- C. Handling Missing Value ------------------------------------------
judul("2C. HANDLING MISSING VALUE", "-")
mask_mp = df2["MINIMUM_PAYMENTS"].isna()
print("[Analisis pola missing] Apakah missing-nya acak (MCAR)?")
banding = df2.groupby(mask_mp)[["BALANCE", "PURCHASES", "PAYMENTS", "CREDIT_LIMIT"]].median().round(2)
banding.index = ["MIN_PAYMENTS terisi", "MIN_PAYMENTS kosong"]
print(banding.to_string())
print(f"""
INTERPRETASI: baris yang MINIMUM_PAYMENTS-nya kosong punya median BALANCE
{df2.loc[mask_mp,'BALANCE'].median():.2f} vs {df2.loc[~mask_mp,'BALANCE'].median():.2f}
dan median PAYMENTS {df2.loc[mask_mp,'PAYMENTS'].median():.2f} vs {df2.loc[~mask_mp,'PAYMENTS'].median():.2f}.
Artinya missing-nya TIDAK acak: itu rekening tidak aktif / dorman.
Konsekuensi: mengisi dengan MEAN ({df2.MINIMUM_PAYMENTS.mean():.2f}) sangat menyesatkan
karena menyuntik nilai besar ke rekening bersaldo hampir nol.""")

print(f"\nPerbandingan mean vs median MINIMUM_PAYMENTS:")
print(f"  mean   = {df2.MINIMUM_PAYMENTS.mean():.2f}   <- terseret outlier (skew={df2.MINIMUM_PAYMENTS.skew():.1f})")
print(f"  median = {df2.MINIMUM_PAYMENTS.median():.2f}   <- dipakai")

df2_clean = df2.copy()
# Flag indikator: menyimpan informasi 'dulunya kosong' agar tidak hilang
df2_clean["MINIMUM_PAYMENTS_was_missing"] = mask_mp.astype(int)
df2_clean["MINIMUM_PAYMENTS"] = df2_clean["MINIMUM_PAYMENTS"].fillna(df2_clean["MINIMUM_PAYMENTS"].median())
df2_clean["CREDIT_LIMIT"] = df2_clean["CREDIT_LIMIT"].fillna(df2_clean["CREDIT_LIMIT"].median())

print("\nSetelah imputasi median:")
cek_missing(df2_clean, "Credit Card")

# ---- D. Deteksi Outlier --------------------------------------------------
judul("2D. DETEKSI OUTLIER (Z-Score & IQR)", "-")
_, mz2, mi2 = deteksi_outlier(df2_clean, kol2, "Credit Card")
print("""
INTERPRETASI: kalau semua baris outlier IQR dibuang, DUA PERTIGA data hilang.
Ini karena distribusinya sangat menceng kanan (skew PURCHASES = 8.1,
MINIMUM_PAYMENTS = 13.6), bukan karena datanya rusak. Nasabah dengan transaksi
besar itu nyata dan justru penting untuk segmentasi.
=> Metode DROP ditolak. Dipakai CAPPING.""")

# ---- E. Handling Outlier -------------------------------------------------
judul("2E. HANDLING OUTLIER", "-")
print("Strategi : capping IQR (winsorizing) pada seluruh kolom numerik.")
df2_final = capping_iqr(df2_clean, kol2)
n_ubah2 = (df2_final[kol2] != df2_clean[kol2]).sum().sum()
print(f"Jumlah nilai yang digeser ke batas : {n_ubah2}")
print(f"Jumlah baris                       : {len(df2_final)} (tidak ada yang dibuang)")

kol2_plot = ["BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "CASH_ADVANCE",
             "PAYMENTS", "MINIMUM_PAYMENTS", "CREDIT_LIMIT"]
boxplot_before_after(df2_clean, df2_final, kol2_plot,
                     "Dataset 2 - Credit Card (kolom moneter, skala symlog)",
                     "02_creditcard_boxplot.png", log=True)
print("\nPerbandingan statistik sebelum vs sesudah capping:")
print(pd.concat([df2_clean[kol2_plot].describe().loc[["mean", "std", "max"]].T.add_suffix("_sebelum"),
                 df2_final[kol2_plot].describe().loc[["mean", "std", "max"]].T.add_suffix("_sesudah")],
                axis=1).round(1).to_string())
df2_final.to_csv(os.path.join(OUT_DIR, "creditcard_clean.csv"), index=False)
ringkasan_akhir.append(["Credit Card", df2.shape, df2_final.shape, 314, 0, n_ubah2])


# =============================================================================
# DATASET 3 : HEART FAILURE PREDICTION
# =============================================================================
judul("DATASET 3 : HEART FAILURE PREDICTION")

# ---- A. Read Data --------------------------------------------------------
df3 = pd.read_csv(os.path.join(DATA_DIR, "heart.csv"))
print("Shape awal :", df3.shape)
print("\n5 baris pertama:")
print(df3.head().to_string())
print("\nInfo tipe data:")
print(df3.dtypes.to_string())

kol3_kontinu = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
kol3_kategori = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
kol3_biner = ["FastingBS", "HeartDisease"]
print(f"\nKolom kontinu  : {kol3_kontinu}")
print(f"Kolom kategori : {kol3_kategori}")
print(f"Kolom biner    : {kol3_biner}  <- TIDAK diuji outlier (0/1 tidak punya outlier)")

# ---- B. Cek Missing Value -----------------------------------------------
judul("3B. CEK MISSING VALUE", "-")
cek_missing(df3, "Heart")

print("\n[Cek missing TERSELUBUNG: nilai 0 yang mustahil secara medis]")
print(f"  Cholesterol == 0 : {(df3.Cholesterol == 0).sum()} baris "
      f"({(df3.Cholesterol == 0).mean() * 100:.1f}%)")
print(f"  RestingBP   == 0 : {(df3.RestingBP == 0).sum()} baris")
print("""
INTERPRETASI: isnull().sum() memberi hasil 0 untuk SEMUA kolom, sehingga kalau
berhenti di situ kita akan menyimpulkan 'data bersih'. Padahal kadar kolesterol
0 mg/dl dan tekanan darah istirahat 0 mmHg mustahil pada orang hidup. Nilai 0 di
sini adalah kode untuk 'tidak diukur' = missing value yang menyamar.""")

print("\n[Apakah missing terselubung ini acak?]")
tab = pd.crosstab(df3.Cholesterol == 0, df3.HeartDisease)
tab.index = ["Cholesterol terukur", "Cholesterol = 0"]
tab.columns = ["Sehat (0)", "Sakit (1)"]
print(tab.to_string())
print(f"""
Dari {(df3.Cholesterol == 0).sum()} baris berkolesterol 0,
{tab.loc['Cholesterol = 0', 'Sakit (1)']} di antaranya pasien SAKIT
({tab.loc['Cholesterol = 0', 'Sakit (1)'] / tab.loc['Cholesterol = 0'].sum() * 100:.0f}%),
jauh di atas proporsi sakit keseluruhan ({df3.HeartDisease.mean() * 100:.0f}%).
Missing-nya berkorelasi dengan label target. Jadi dropna() akan membuang
{tab.loc['Cholesterol = 0', 'Sakit (1)']} kasus positif dan membuat data bias.""")

# ---- C. Handling Missing Value ------------------------------------------
judul("3C. HANDLING MISSING VALUE", "-")
df3_clean = df3.copy()
df3_clean["Cholesterol_was_zero"] = (df3_clean.Cholesterol == 0).astype(int)
df3_clean["Cholesterol"] = df3_clean["Cholesterol"].replace(0, np.nan)
df3_clean["RestingBP"] = df3_clean["RestingBP"].replace(0, np.nan)
print("Langkah 1: nilai 0 pada Cholesterol & RestingBP diubah jadi NaN.")
cek_missing(df3_clean, "Heart (setelah 0 -> NaN)")

print(f"""
Langkah 2: bandingkan statistik SEBELUM dan SESUDAH nol dianggap missing.
  median Cholesterol dengan nol dihitung  : {df3.Cholesterol.median():.1f}  (BIAS)
  median Cholesterol dengan nol = NaN     : {df3_clean.Cholesterol.median():.1f}  (benar)
  mean   Cholesterol dengan nol dihitung  : {df3.Cholesterol.mean():.1f}  (BIAS)
  mean   Cholesterol dengan nol = NaN     : {df3_clean.Cholesterol.mean():.1f}  (benar)
Selisih mean ~{abs(df3.Cholesterol.mean() - df3_clean.Cholesterol.mean()) / df3_clean.Cholesterol.mean() * 100:.0f}%.
Ini bukti kenapa urutan 'perbaiki missing dulu, baru hitung statistik' itu wajib.""")

print("\nLangkah 3: imputasi dengan MEDIAN (bukan mean, karena ada outlier).")
for col in ["Cholesterol", "RestingBP"]:
    df3_clean[col] = df3_clean[col].fillna(df3_clean[col].median())
cek_missing(df3_clean, "Heart (setelah imputasi)")

print("\nLangkah 4: cek missing pada kolom kategori (imputasi modus jika ada).")
for col in df3_clean.select_dtypes(include=["object"]).columns:
    n = df3_clean[col].isna().sum()
    if n > 0:
        df3_clean[col] = df3_clean[col].fillna(df3_clean[col].mode()[0])
        print(f"  {col}: {n} NaN diisi modus '{df3_clean[col].mode()[0]}'")
print("  Tidak ada NaN pada kolom kategori -> tidak ada imputasi modus.")

# ---- D. Deteksi Outlier --------------------------------------------------
judul("3D. DETEKSI OUTLIER (Z-Score & IQR)", "-")
print(">> Perbandingan: deteksi SEBELUM vs SESUDAH nol diperbaiki\n")
print("(a) Kalau nol dibiarkan (cara naif):")
_, _, mi3_naif = deteksi_outlier(df3, kol3_kontinu, "Heart - nol dibiarkan")
print("\n(b) Setelah nol diperbaiki jadi NaN lalu diimputasi:")
_, mz3, mi3 = deteksi_outlier(df3_clean, kol3_kontinu, "Heart - sudah diperbaiki")
print(f"""
INTERPRETASI: outlier IQR pada Cholesterol turun dari {mi3_naif['Cholesterol'] if isinstance(mi3_naif, pd.Series) else (mi3_naif['Cholesterol']).sum()} menjadi {mi3['Cholesterol'].sum()}.
Sebagian besar 'outlier' pada langkah (a) sebenarnya adalah missing value yang
menyamar, bukan outlier. Kalau langsung dibuang, kita membuang data sehat.""")

# ---- E. Handling Outlier -------------------------------------------------
judul("3E. HANDLING OUTLIER", "-")
print("Strategi : capping IQR pada kolom kontinu (Age dikecualikan, tidak ada outlier).")
kol3_cap = ["RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
df3_final = capping_iqr(df3_clean, kol3_cap)
n_ubah3 = (df3_final[kol3_cap] != df3_clean[kol3_cap]).sum().sum()
print(f"Jumlah nilai yang digeser ke batas : {n_ubah3}")
print(f"Jumlah baris                       : {len(df3_final)} (tidak ada yang dibuang)")
boxplot_before_after(df3_clean, df3_final, kol3_kontinu,
                     "Dataset 3 - Heart Failure Prediction",
                     "03_heart_boxplot.png")
df3_final.to_csv(os.path.join(OUT_DIR, "heart_clean.csv"), index=False)
ringkasan_akhir.append(["Heart", df3.shape, df3_final.shape,
                        int((df3.Cholesterol == 0).sum() + (df3.RestingBP == 0).sum()), 0, n_ubah3])


# =============================================================================
# RINGKASAN AKHIR
# =============================================================================
judul("RINGKASAN HASIL PREPROCESSING KETIGA DATASET")
ring = pd.DataFrame(ringkasan_akhir, columns=[
    "Dataset", "Shape awal", "Shape akhir", "Missing ditangani",
    "Duplikat dibuang", "Nilai di-capping"])
print(ring.to_string(index=False))
print(f"\nFile bersih tersimpan di folder '{OUT_DIR}/'")
print(f"Gambar boxplot tersimpan di folder '{FIG_DIR}/'")
print("\nSELESAI.")