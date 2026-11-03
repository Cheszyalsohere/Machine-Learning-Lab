# =============================================================
#  TUGAS REGRESI - MACHINE LEARNING (Pertemuan 10)
#  File    : regresi_kelompok5.py
#  Kelompok: 5 (TI-C5) - Muhammad Irfan Nuha (434241079)
#                        Bunga Lala Sal Sabila (434241066)
#  Dataset : Concrete Compressive Strength (1.030 sampel beton)
#  Target  : concrete_compressive_strength (kuat tekan beton, MPa)
#  Metode  : Gradient Boosting Regressor
#            (Linear Regression dipakai sebagai pembanding)
#
#  Alur:
#    Input -> Preprocessing (missing, duplikat, outlier)
#    -> Eksplorasi korelasi -> Split 80/20
#    -> Linear Regression (pembanding) & Gradient Boosting
#    -> Evaluasi (MAE, MSE, RMSE, R2) -> Feature importance
#    -> Contoh prediksi
# =============================================================

import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)
TARGET = "concrete_compressive_strength"


def judul(teks):
    print("\n" + "=" * 64)
    print(f"  {teks}")
    print("=" * 64)


def tampilkan(nama_file):
    plt.savefig(os.path.join(OUT_DIR, nama_file), dpi=110, bbox_inches="tight")
    plt.show()
    plt.close()


def evaluasi(y_asli, y_pred):
    """Hitung MAE, MSE, RMSE, dan R2."""
    mse = mean_squared_error(y_asli, y_pred)
    return {"MAE": mean_absolute_error(y_asli, y_pred), "MSE": mse,
            "RMSE": np.sqrt(mse), "R2": r2_score(y_asli, y_pred)}


def main():
    # ---------------- 1. INPUT ----------------
    judul("1. INPUT DATA")
    df = pd.read_csv(os.path.join(BASE_DIR, "concrete_data.csv"))
    df.columns = df.columns.str.strip()          # ada spasi di nama kolom 'fine_aggregate '
    print(f"Ukuran data : {df.shape[0]} baris x {df.shape[1]} kolom")
    print(f"Target      : {TARGET} (MPa)")
    print(df.head().to_string(index=False))

    # ---------------- 2. PREPROCESSING ----------------
    judul("2. PREPROCESSING")
    print(f">> Missing value : {int(df.isna().sum().sum())} sel")
    dup = int(df.duplicated().sum())
    print(f">> Duplikasi     : {dup} baris")
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"   -> duplikat dihapus, sisa {len(df)} baris")

    print(">> Outlier (IQR):")
    for c in df.columns:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        n = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
        if n:
            print(f"     {c:<30}: {n} baris")
    print("   -> tidak dibuang: nilainya wajar untuk campuran beton (mis. umur")
    print("      beton 180-365 hari), dan Gradient Boosting tahan terhadap outlier")

    # ---------------- 3. EKSPLORASI ----------------
    judul("3. KORELASI FITUR DENGAN TARGET")
    korelasi = df.corr()[TARGET].drop(TARGET).sort_values(ascending=False)
    for f, r in korelasi.items():
        print(f"   {f:<20}: {r:+.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax[0],
                annot_kws={"size": 7}, cbar=False)
    ax[0].set_title("Matriks korelasi")
    ax[1].scatter(df["age"], df[TARGET], s=10, alpha=0.5, color="#3d7cc9")
    rata_umur = df.groupby("age")[TARGET].mean()
    ax[1].plot(rata_umur.index, rata_umur.values, "r-o", ms=4, label="rata-rata per umur")
    ax[1].set_xlabel("age (hari)")
    ax[1].set_ylabel(f"{TARGET} (MPa)")
    ax[1].set_title("Umur beton vs kuat tekan (hubungan tidak linear)")
    ax[1].legend()
    plt.tight_layout()
    tampilkan("reg_1_eksplorasi.png")

    # ---------------- 4. SPLIT DATA ----------------
    judul("4. SPLIT DATA 80% TRAINING / 20% TESTING")
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print(f"   Training : {len(X_train)} baris")
    print(f"   Testing  : {len(X_test)} baris")

    # ---------------- 5. PEMODELAN ----------------
    judul("5. PEMODELAN")
    linear = LinearRegression().fit(X_train, y_train)
    print("   Linear Regression (pembanding) selesai dilatih")
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3,
                                    random_state=42)
    gbr.fit(X_train, y_train)
    print("   Gradient Boosting Regressor selesai dilatih")
    print("     n_estimators=100 | learning_rate=0.1 | max_depth=3")

    # ---------------- 6. EVALUASI ----------------
    judul("6. EVALUASI MODEL (data testing)")
    hasil = {
        "Linear Regression": evaluasi(y_test, linear.predict(X_test)),
        "Gradient Boosting": evaluasi(y_test, gbr.predict(X_test)),
    }
    tabel = pd.DataFrame(hasil).T
    print(tabel.round(4).to_string())
    r2_train = r2_score(y_train, gbr.predict(X_train))
    print(f"\n   R2 Gradient Boosting data training: {r2_train:.4f}")
    tabel.to_csv(os.path.join(OUT_DIR, "evaluasi.csv"))

    pred_lin, pred_gbr = linear.predict(X_test), gbr.predict(X_test)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for a, p, nama in [(ax[0], pred_lin, "Linear Regression"), (ax[1], pred_gbr, "Gradient Boosting")]:
        a.scatter(y_test, p, s=14, alpha=0.6, color="#3d7cc9")
        a.plot([0, 85], [0, 85], "r--", label="prediksi = aktual")
        a.set_xlabel("Kuat tekan aktual (MPa)")
        a.set_title(f"{nama} (R2 = {r2_score(y_test, p):.3f})")
        a.legend()
    ax[0].set_ylabel("Kuat tekan prediksi (MPa)")
    plt.tight_layout()
    tampilkan("reg_2_aktual_vs_prediksi.png")

    # ---------------- 7. FEATURE IMPORTANCE ----------------
    judul("7. FEATURE IMPORTANCE GRADIENT BOOSTING")
    penting = pd.Series(gbr.feature_importances_, index=X.columns).sort_values(ascending=False)
    for f, v in penting.items():
        print(f"   {f:<20}: {v * 100:5.1f}%")

    plt.figure(figsize=(9, 4.5))
    penting.iloc[::-1].mul(100).plot(kind="barh", color="#3d7cc9")
    plt.xlabel("Kontribusi (%)")
    plt.title("Feature importance Gradient Boosting")
    plt.tight_layout()
    tampilkan("reg_3_feature_importance.png")

    # ---------------- 8. CONTOH PREDIKSI ----------------
    judul("8. CONTOH PREDIKSI (5 data testing)")
    contoh = X_test.head(5)[["cement", "water", "age"]].copy()
    contoh["Aktual"] = y_test.head(5).values
    contoh["Prediksi"] = pred_gbr[:5].round(2)
    contoh["Selisih"] = (contoh["Aktual"] - contoh["Prediksi"]).round(2)
    print(contoh.to_string(index=False))

    campuran = pd.DataFrame([{"cement": 350, "blast_furnace_slag": 100, "fly_ash": 0, "water": 180,
                              "superplasticizer": 8, "coarse_aggregate": 950,
                              "fine_aggregate": 750, "age": 28}])[X.columns]
    print("\n   Campuran baru (semen 350, slag 100, air 180, superplasticizer 8, umur 28 hari)")
    print(f"   Prediksi kuat tekan: {gbr.predict(campuran)[0]:.2f} MPa")

    X_test.assign(Aktual=y_test.values, Prediksi=pred_gbr.round(2)).to_csv(
        os.path.join(OUT_DIR, "hasil_prediksi.csv"), index=False)
    print("\nHasil disimpan -> output/hasil_prediksi.csv, output/evaluasi.csv")


if __name__ == "__main__":
    main()
