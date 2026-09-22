# =============================================================
#  TUGAS IMBALANCED DATA - MACHINE LEARNING (Pertemuan 6)
#  File   : imbalanced_data_434241079.py
#  NIM    : 434241079
#  Dataset: heart_preprocessed.csv (hasil preprocessing Pertemuan 2:
#           handling missing value + capping outlier), target = HeartDisease
#  Metode : SMOTE, SMOTE-ENN, ROS (Random Over Sampling),
#           dan RUS (Random Under Sampling)
#
#  Tujuan: menyeimbangkan jumlah kelas 0 dan 1 pada target,
#          lalu membandingkan hasil tiap metode.
# =============================================================

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)
TARGET   = "HeartDisease"
LABELS   = {0: "Tidak Sakit", 1: "Sakit"}


def dist(y):
    """Kembalikan dict jumlah tiap kelas (urut label)."""
    return y.value_counts().sort_index().to_dict()


def plot_dist(y, judul, fname):
    vc = y.value_counts().sort_index()
    nama_kelas = [LABELS[i] for i in vc.index]
    plt.figure(figsize=(5, 4))
    sns.barplot(x=nama_kelas, y=vc.values, hue=nama_kelas,
                palette="coolwarm", legend=False)
    for i, v in enumerate(vc.values):
        plt.text(i, v, str(v), ha="center", va="bottom", fontsize=10)
    plt.xlabel("Status"); plt.ylabel("Jumlah"); plt.title(judul)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=120); plt.close()


def main():
    df = pd.read_csv(os.path.join(BASE_DIR, "heart_preprocessed.csv"))
    df.columns = df.columns.str.strip()
    print("=" * 60)
    print("  IMBALANCED DATA - heart_preprocessed.csv")
    print("=" * 60)
    print(f"Ukuran data : {df.shape[0]} baris x {df.shape[1]} kolom")

    # Encode kolom kategorikal -> angka (metode resampling butuh numerik)
    for kol in df.select_dtypes(include="object").columns:
        df[kol] = LabelEncoder().fit_transform(df[kol])

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    print("\nDistribusi AWAL (sebelum penyeimbangan):")
    for k, v in dist(y).items():
        print(f"  Kelas {k} ({LABELS[k]:<11}): {v}")
    plot_dist(y, "Distribusi Awal (Sebelum)", "dist_awal.png")

    # Definisikan 4 metode
    metode = {
        "SMOTE":    SMOTE(random_state=42),
        "SMOTE-ENN": SMOTEENN(random_state=42),
        "ROS":      RandomOverSampler(random_state=42),
        "RUS":      RandomUnderSampler(random_state=42),
    }
    fmap = {"SMOTE": "dist_smote.png", "SMOTE-ENN": "dist_smoteenn.png",
            "ROS": "dist_ros.png", "RUS": "dist_rus.png"}

    ringkas = []
    for nama, sampler in metode.items():
        X_res, y_res = sampler.fit_resample(X, y)
        d = dist(y_res)
        print(f"\nDistribusi SESUDAH {nama}:")
        for k, v in d.items():
            print(f"  Kelas {k} ({LABELS[k]:<11}): {v}")
        print(f"  Total baris: {len(y_res)}")
        plot_dist(y_res, f"Distribusi Sesudah {nama}", fmap[nama])
        ringkas.append({"Metode": nama, "Kelas 0": d.get(0, 0),
                        "Kelas 1": d.get(1, 0), "Total": len(y_res)})

    # Tabel ringkasan
    awal = dist(y)
    ringkas.insert(0, {"Metode": "AWAL", "Kelas 0": awal[0],
                       "Kelas 1": awal[1], "Total": len(y)})
    tabel = pd.DataFrame(ringkas)
    print("\n" + "=" * 60)
    print("  RINGKASAN JUMLAH DATA TIAP METODE")
    print("=" * 60)
    print(tabel.to_string(index=False))

    print("\nGrafik distribusi tersimpan di folder output/")


if __name__ == "__main__":
    main()
