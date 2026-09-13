# =============================================================
#  TUGAS EKSTRAKSI FITUR - MACHINE LEARNING (Pertemuan 5)
#  File   : ekstraksi_fitur_pca_optimal_434241079.py
#  NIM    : 434241079
#  Metode : PCA dengan jumlah komponen OPTIMAL
#  Dataset: heart.csv   (target = HeartDisease)
#
#  n optimal = jumlah komponen terkecil yang varians kumulatifnya
#  sudah mencapai ambang 95% (informasi yang dipertahankan >= 95%).
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
AMBANG   = 0.95


def main():
    df = pd.read_csv(os.path.join(BASE_DIR, "heart.csv"))
    print("=" * 60)
    print("  EKSTRAKSI FITUR - PCA (n komponen OPTIMAL, ambang 95%)")
    print("=" * 60)
    print(f"Dataset : heart.csv  ({df.shape[0]} baris x {df.shape[1]} kolom)")

    X = df.drop(columns=[TARGET]).copy()
    y = df[TARGET]
    for kol in X.select_dtypes(include="object").columns:
        X[kol] = LabelEncoder().fit_transform(X[kol])
    print(f"Jumlah fitur setelah encoding : {X.shape[1]}")

    X_scaled = StandardScaler().fit_transform(X)

    # 1) PCA penuh untuk melihat varians tiap komponen
    pca_full = PCA().fit(X_scaled)
    ev  = pca_full.explained_variance_ratio_
    cum = np.cumsum(ev)

    print(f"\n{'Komponen':<10}{'Varians':>10}{'Kumulatif':>12}")
    print("-" * 32)
    for i, (e, c) in enumerate(zip(ev, cum), 1):
        print(f"PC{i:<8}{e*100:>9.2f}%{c*100:>11.2f}%")

    # 2) Tentukan n optimal (kumulatif >= 95%)
    n_optimal = int(np.argmax(cum >= AMBANG) + 1)
    print(f"\nn komponen optimal (>= {AMBANG*100:.0f}% varians) : {n_optimal} "
          f"dari {X.shape[1]} fitur asli")
    print(f"Varians dipertahankan : {cum[n_optimal-1]*100:.2f}%")

    # 3) PCA dengan n optimal + simpan hasil
    pca = PCA(n_components=n_optimal)
    X_pca = pca.fit_transform(X_scaled)
    kolom = [f"PC{i}" for i in range(1, n_optimal+1)]
    df_pca = pd.DataFrame(X_pca, columns=kolom)
    df_pca[TARGET] = y.values
    df_pca.to_csv(os.path.join(BASE_DIR, "hasil_pca_optimal.csv"), index=False)
    print("\n5 baris pertama hasil PCA optimal:")
    print(df_pca.head().to_string(index=False))

    # 4) Grafik scree / varians kumulatif
    plt.figure(figsize=(8, 5))
    x = range(1, len(cum)+1)
    plt.bar(x, ev*100, alpha=0.6, label="Varians per komponen")
    plt.plot(x, cum*100, "o-", color="crimson", label="Varians kumulatif")
    plt.axhline(AMBANG*100, ls="--", color="gray", label=f"Ambang {AMBANG*100:.0f}%")
    plt.axvline(n_optimal, ls=":", color="green", label=f"n optimal = {n_optimal}")
    plt.xlabel("Jumlah Komponen Utama")
    plt.ylabel("Varians Dijelaskan (%)")
    plt.title("Scree Plot & Varians Kumulatif - heart.csv")
    plt.xticks(list(x))
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pca_optimal_scree.png"), dpi=120)
    plt.close()

    print("\nHasil disimpan -> hasil_pca_optimal.csv")
    print("Grafik         -> output/pca_optimal_scree.png")


if __name__ == "__main__":
    main()
