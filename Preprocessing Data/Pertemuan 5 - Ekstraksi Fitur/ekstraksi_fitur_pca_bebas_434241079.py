# =============================================================
#  TUGAS EKSTRAKSI FITUR - MACHINE LEARNING (Pertemuan 5)
#  File   : ekstraksi_fitur_pca_bebas_434241079.py
#  NIM    : 434241079
#  Metode : PCA dengan jumlah komponen BEBAS (n_components = 2)
#  Dataset: heart.csv   (target = HeartDisease)
#
#  n = 2 dipilih bebas agar hasil PCA bisa divisualisasikan
#  dalam scatter plot 2 dimensi (PC1 vs PC2).
# =============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)
TARGET   = "HeartDisease"
N_KOMPONEN = 2


def main():
    df = pd.read_csv(os.path.join(BASE_DIR, "heart.csv"))
    print("=" * 60)
    print("  EKSTRAKSI FITUR - PCA (n komponen BEBAS = 2)")
    print("=" * 60)
    print(f"Dataset : heart.csv  ({df.shape[0]} baris x {df.shape[1]} kolom)")

    # Pisahkan fitur (X) dan target (y)
    X = df.drop(columns=[TARGET]).copy()
    y = df[TARGET]

    # Encode kolom kategorikal menjadi angka (PCA hanya menerima angka)
    for kol in X.select_dtypes(include="object").columns:
        X[kol] = LabelEncoder().fit_transform(X[kol])
    print(f"Jumlah fitur setelah encoding : {X.shape[1]}")

    # Standarisasi (PCA butuh data pada skala yang sama)
    X_scaled = StandardScaler().fit_transform(X)

    # PCA dengan 2 komponen
    pca = PCA(n_components=N_KOMPONEN)
    X_pca = pca.fit_transform(X_scaled)

    ev = pca.explained_variance_ratio_
    print("\nExplained Variance Ratio:")
    for i, v in enumerate(ev, 1):
        print(f"  PC{i} : {v*100:.2f}%")
    print(f"Total Variance Explained ({N_KOMPONEN} komponen): {ev.sum()*100:.2f}%")
    print(f"Informasi yang hilang: {(1-ev.sum())*100:.2f}%")

    # Simpan hasil ke DataFrame + CSV
    df_pca = pd.DataFrame(X_pca, columns=[f"PC{i}" for i in range(1, N_KOMPONEN+1)])
    df_pca[TARGET] = y.values
    df_pca.to_csv(os.path.join(BASE_DIR, "hasil_pca_bebas.csv"), index=False)
    print("\n5 baris pertama hasil PCA:")
    print(df_pca.head().to_string(index=False))

    # Visualisasi scatter PC1 vs PC2
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(df_pca["PC1"], df_pca["PC2"], c=df_pca[TARGET], cmap="coolwarm", alpha=0.7)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA (2 Komponen) - heart.csv")
    plt.colorbar(sc, label="HeartDisease")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pca_bebas_scatter.png"), dpi=120)
    plt.close()

    print("\nHasil disimpan -> hasil_pca_bebas.csv")
    print("Grafik         -> output/pca_bebas_scatter.png")


if __name__ == "__main__":
    main()
