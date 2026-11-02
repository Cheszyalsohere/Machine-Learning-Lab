# =============================================================
#  TUGAS REGRESI - MACHINE LEARNING (Pertemuan 10)
#  File    : regresi_kelompok5.py
#  Kelompok: 5 (TI-C5) - Muhammad Irfan Nuha (434241079)
#                        Bunga Lala Sal Sabila (434241066)
#  Dataset : Airline Passenger Satisfaction (dataset UTS kelompok)
#  Target  : Arrival Delay in Minutes (keterlambatan tiba, menit)
#  Metode  : Linear Regression
#
#  Alur:
#    Input -> Preprocessing (missing, duplikat, outlier, encoding)
#    -> Split 80/20 -> Linear Regression -> Evaluasi (MAE, MSE,
#    RMSE, R2) -> Interpretasi koefisien -> Contoh prediksi
# =============================================================

import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)
TARGET = "Arrival Delay in Minutes"


def judul(teks):
    print("\n" + "=" * 64)
    print(f"  {teks}")
    print("=" * 64)


def tampilkan(nama_file):
    plt.savefig(os.path.join(OUT_DIR, nama_file), dpi=110, bbox_inches="tight")
    plt.show()
    plt.close()


def main():
    # ---------------- 1. INPUT ----------------
    judul("1. INPUT DATA")
    df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
    df = df.drop(columns=[c for c in ["Unnamed: 0", "id"] if c in df.columns])
    print(f"Ukuran data : {df.shape[0]} baris x {df.shape[1]} kolom")
    print(f"Target      : {TARGET}")
    print(df[["Gender", "Class", "Flight Distance", "Departure Delay in Minutes", TARGET]]
          .head().to_string(index=False))

    # ---------------- 2. PREPROCESSING ----------------
    judul("2. PREPROCESSING")
    miss = df.isna().sum()
    print(">> Missing value:")
    for k, v in miss[miss > 0].items():
        print(f"     {k}: {v}")
    # missing ada di kolom TARGET -> baris dihapus (target tidak boleh dikarang/diisi)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    print(f"   -> baris dengan target kosong dihapus, sisa {len(df)} baris")

    dup = int(df.duplicated().sum())
    print(f">> Duplikasi data: {dup} baris")
    df = df.drop_duplicates().reset_index(drop=True)

    print(">> Outlier (IQR):")
    for c in ["Flight Distance", "Departure Delay in Minutes", TARGET]:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        n = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
        print(f"     {c:<28}: {n:>6} baris (maks {df[c].max():.0f})")
    print("   -> tidak dibuang: keterlambatan panjang adalah kejadian nyata,")
    print("      bukan salah input, dan tetap mengikuti pola linear delay")

    # one-hot encoding untuk kolom kategorikal (Linear Regression butuh angka)
    kategori = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=kategori, drop_first=True, dtype=int)
    print(f">> One-hot encoding: {kategori}")
    print(f"   -> total {df.shape[1] - 1} fitur")

    # ---------------- 3. KORELASI ----------------
    judul("3. KORELASI FITUR DENGAN TARGET (5 teratas)")
    korelasi = df.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    for f, r in korelasi.head(5).items():
        print(f"   {f:<34}: {r:+.3f}")

    plt.figure(figsize=(8, 5))
    plt.scatter(df["Departure Delay in Minutes"], df[TARGET], s=4, alpha=0.3, color="#3d7cc9")
    plt.xlabel("Departure Delay in Minutes")
    plt.ylabel(TARGET)
    plt.title("Keterlambatan berangkat vs keterlambatan tiba")
    tampilkan("reg_1_scatter_delay.png")

    # ---------------- 4. SPLIT DATA ----------------
    judul("4. SPLIT DATA 80% TRAINING / 20% TESTING")
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print(f"   Training : {len(X_train)} baris")
    print(f"   Testing  : {len(X_test)} baris")

    # ---------------- 5. LINEAR REGRESSION ----------------
    judul("5. MODEL LINEAR REGRESSION")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"   Intercept : {model.intercept_:.4f}")
    koef = pd.Series(model.coef_, index=X.columns).sort_values(key=abs, ascending=False)
    print("   Koefisien 5 fitur paling berpengaruh:")
    for f, k in koef.head(5).items():
        print(f"     {f:<34}: {k:+.4f}")

    # ---------------- 6. EVALUASI ----------------
    judul("6. EVALUASI MODEL")
    pred_train = model.predict(X_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)
    print(f"   MAE  : {mae:.4f} menit")
    print(f"   MSE  : {mse:.4f}")
    print(f"   RMSE : {rmse:.4f} menit")
    print(f"   R2   : {r2:.4f}  (data training: {r2_score(y_train, pred_train):.4f})")
    pd.DataFrame({"Metrik": ["MAE", "MSE", "RMSE", "R2"], "Nilai": [mae, mse, rmse, r2]}).to_csv(
        os.path.join(OUT_DIR, "evaluasi.csv"), index=False)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].scatter(y_test, pred, s=5, alpha=0.3, color="#3d7cc9")
    batas = [0, max(y_test.max(), pred.max())]
    ax[0].plot(batas, batas, "r--", label="prediksi = aktual")
    ax[0].set_xlabel("Arrival Delay aktual (menit)")
    ax[0].set_ylabel("Arrival Delay prediksi (menit)")
    ax[0].set_title(f"Aktual vs Prediksi (R2 = {r2:.3f})")
    ax[0].legend()
    residu = y_test - pred
    ax[1].hist(residu, bins=80, range=(-80, 80), color="#8fb3e0", edgecolor="white")
    ax[1].axvline(0, color="red", ls="--")
    ax[1].set_xlabel("Residu = aktual - prediksi (menit)")
    ax[1].set_ylabel("Jumlah data")
    ax[1].set_title("Sebaran residu")
    plt.tight_layout()
    tampilkan("reg_2_aktual_vs_prediksi.png")

    plt.figure(figsize=(9, 5))
    top = koef.head(10).iloc[::-1]
    plt.barh(top.index, top.values, color=["#d66b6b" if v > 0 else "#5b8fd6" for v in top.values])
    plt.axvline(0, color="black", lw=0.8)
    plt.xlabel("Koefisien (menit per satuan fitur)")
    plt.title("10 koefisien Linear Regression terbesar")
    plt.tight_layout()
    tampilkan("reg_3_koefisien.png")

    # ---------------- 7. CONTOH PREDIKSI ----------------
    judul("7. CONTOH PREDIKSI (5 data testing)")
    contoh = pd.DataFrame({
        "Departure Delay": X_test["Departure Delay in Minutes"].values[:5],
        "Arrival Delay asli": y_test.values[:5],
        "Prediksi": pred[:5].round(1),
    })
    contoh["Selisih"] = (contoh["Arrival Delay asli"] - contoh["Prediksi"]).round(1)
    print(contoh.to_string(index=False))

    hasil = X_test.assign(**{"Arrival Delay asli": y_test.values, "Prediksi": pred.round(2)})
    hasil.to_csv(os.path.join(OUT_DIR, "hasil_prediksi.csv"), index=False)
    print("\nHasil disimpan -> output/hasil_prediksi.csv, output/evaluasi.csv")


if __name__ == "__main__":
    main()
