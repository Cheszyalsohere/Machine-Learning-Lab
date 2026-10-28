# =============================================================
#  TUGAS CLUSTERING - MACHINE LEARNING (Pertemuan 9)
#  File    : kmodes_student_performance.py
#  Kelompok: 5 (TI-C5, kelompok ganjil -> K-Modes)
#            Muhammad Irfan Nuha (434241079)
#            Bunga Lala Sal Sabila (434241066)
#  Dataset : Student Performance Factors (6.607 siswa, 20 kolom)
#
#  Alur:
#    Input -> Preprocessing (missing, duplikat, outlier)
#    -> Diskretisasi kolom numerik menjadi kategori (syarat K-Modes)
#    -> Elbow Method (cost) untuk menentukan k
#    -> K-Modes -> Evaluasi (Silhouette, jarak Hamming)
#    -> Profil tiap cluster (termasuk rata-rata Exam_Score)
# =============================================================

import os
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)
NILAI = "Exam_Score"          # tidak ikut di-cluster, dipakai untuk membaca hasil
K_MAKS = 10


def judul(teks):
    print("\n" + "=" * 64)
    print(f"  {teks}")
    print("=" * 64)


def tampilkan(nama_file):
    plt.savefig(os.path.join(OUT_DIR, nama_file), dpi=110, bbox_inches="tight")
    plt.show()
    plt.close()


def titik_siku(nilai):
    """Elbow otomatis: titik dengan jarak terjauh dari garis k=1 ke k=K_MAKS."""
    y = np.array(nilai, dtype=float)
    k = np.arange(1, len(y) + 1)
    # normalisasi ke [0, 1] agar skala sumbu k dan cost sebanding
    xn = (k - k[0]) / (k[-1] - k[0])
    yn = (y - y.min()) / (y.max() - y.min())
    # garis dari titik pertama (0, 1) ke terakhir (1, 0): x + y - 1 = 0
    jarak = np.abs(xn + yn - 1) / np.sqrt(2)
    return int(k[np.argmax(jarak)])


def main():
    # ---------------- 1. INPUT ----------------
    judul("1. INPUT DATA")
    df = pd.read_csv(os.path.join(BASE_DIR, "StudentPerformanceFactors.csv"))
    print(f"Ukuran data : {df.shape[0]} baris x {df.shape[1]} kolom")
    kol_kat = df.select_dtypes(include="object").columns.tolist()
    kol_num = [c for c in df.select_dtypes(include="number").columns if c != NILAI]
    print(f"Kategorikal : {len(kol_kat)} kolom | Numerik : {len(kol_num)} kolom + {NILAI}")
    print(df.iloc[:5, :5].to_string(index=False))
    print(f"... (+{df.shape[1] - 5} kolom lainnya)")

    # ---------------- 2. PREPROCESSING ----------------
    judul("2. PREPROCESSING")
    miss = df.isna().sum()
    print(">> Missing value:")
    for k, v in miss[miss > 0].items():
        print(f"     {k:<26}: {v}")
    for c in kol_kat:                               # isi dengan modus
        df[c] = df[c].fillna(df[c].mode()[0])
    print(f"   -> diisi modus, sisa missing = {int(df.isna().sum().sum())}")

    dup = int(df.duplicated().sum())
    print(f">> Duplikasi data: {dup} baris")
    df = df.drop_duplicates().reset_index(drop=True)

    print(">> Outlier (IQR) kolom numerik:")
    for c in kol_num + [NILAI]:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        n = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
        print(f"     {c:<26}: {n}")
    lebih = int((df[NILAI] > 100).sum())
    df[NILAI] = df[NILAI].clip(upper=100)          # nilai ujian maksimal 100
    print(f"   -> {NILAI} > 100 sebanyak {lebih} baris di-cap menjadi 100")
    print("   -> outlier kolom lain tidak dibuang; setelah diskretisasi otomatis")
    print("      masuk kategori Rendah/Tinggi sehingga tidak merusak K-Modes")

    # ---------------- 3. DISKRETISASI ----------------
    judul("3. DISKRETISASI NUMERIK -> KATEGORI (syarat K-Modes)")
    data = df[kol_kat].copy()
    for c in kol_num:
        bins = pd.qcut(df[c], q=3, duplicates="drop", retbins=True)[1]
        label = ["Rendah", "Sedang", "Tinggi"] if len(bins) == 4 else ["Rendah", "Tinggi"]
        data[c] = pd.cut(df[c], bins=bins, labels=label, include_lowest=True).astype(str)
        rentang = ", ".join(f"{l}<={b:g}" for l, b in zip(label, bins[1:]))
        print(f"   {c:<20}: {rentang}")
    print(f"   Fitur untuk clustering: {data.shape[1]} kolom kategorikal "
          f"({NILAI} tidak ikut)")

    # ---------------- 4. ELBOW METHOD ----------------
    judul(f"4. ELBOW METHOD (k = 1 sampai {K_MAKS})")
    cost = []
    for k in range(1, K_MAKS + 1):
        km = KModes(n_clusters=k, init="Huang", n_init=5, random_state=42)
        km.fit(data)
        cost.append(km.cost_)
        print(f"   k = {k:<2}  cost = {km.cost_:10.0f}")
    k_opt = titik_siku(cost)
    print(f"   Titik siku (elbow) : k = {k_opt}")

    plt.figure(figsize=(9, 4.5))
    plt.plot(range(1, K_MAKS + 1), cost, "o-", color="#3d7cc9")
    plt.axvline(k_opt, ls="--", color="crimson", label=f"k terpilih = {k_opt}")
    plt.xlabel("Jumlah cluster (k)")
    plt.ylabel("Cost (jumlah ketidakcocokan)")
    plt.title("Elbow Method K-Modes - Student Performance")
    plt.xticks(range(1, K_MAKS + 1))
    plt.legend()
    tampilkan("kmodes_1_elbow.png")

    # ---------------- 5. K-MODES ----------------
    judul(f"5. K-MODES (k = {k_opt})")
    mulai = time.time()
    km = KModes(n_clusters=k_opt, init="Huang", n_init=10, random_state=42)
    label_cluster = km.fit_predict(data)
    print(f"   Iterasi selesai dalam {time.time() - mulai:.1f} detik, cost = {km.cost_:.0f}")
    df["Cluster"] = label_cluster
    ukuran = df["Cluster"].value_counts().sort_index()
    for c, n in ukuran.items():
        print(f"   Cluster {c}: {n} siswa ({n / len(df) * 100:.1f}%)")

    centroid = pd.DataFrame(km.cluster_centroids_, columns=data.columns)
    centroid.index = [f"Cluster {i}" for i in range(k_opt)]
    centroid.T.to_csv(os.path.join(OUT_DIR, "centroid_kmodes.csv"))

    # ---------------- 6. EVALUASI ----------------
    judul("6. EVALUASI - SILHOUETTE SCORE (jarak Hamming)")
    kode = data.apply(lambda s: s.astype("category").cat.codes)
    skor = silhouette_score(kode, label_cluster, metric="hamming")
    print(f"   Silhouette Score: {skor:.4f}")

    # ---------------- 7. PROFIL CLUSTER ----------------
    judul(f"7. PROFIL CLUSTER (modus tiap fitur + rata-rata {NILAI})")
    beda = [c for c in centroid.columns if centroid[c].nunique() > 1]
    print("   Fitur yang membedakan antar-cluster:")
    print(centroid[beda].T.to_string())
    rata = df.groupby("Cluster")[NILAI].mean().round(2)
    print(f"\n   Rata-rata {NILAI} per cluster:")
    for c, v in rata.items():
        print(f"     Cluster {c}: {v}")

    df.to_csv(os.path.join(OUT_DIR, "hasil_clustering_kmodes.csv"), index=False)

    # grafik ukuran cluster + nilai ujian
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].bar([f"C{i}" for i in ukuran.index], ukuran.values, color="#3d7cc9")
    for i, v in enumerate(ukuran.values):
        ax[0].text(i, v, f"{v}", ha="center", va="bottom", fontsize=9)
    ax[0].set_title("Jumlah siswa per cluster")
    sns.boxplot(x=df["Cluster"].map(lambda c: f"C{c}"), y=df[NILAI], ax=ax[1],
                order=[f"C{i}" for i in range(k_opt)], color="#8fb3e0")
    ax[1].set_xlabel("Cluster")
    ax[1].set_title(f"Sebaran {NILAI} per cluster")
    plt.tight_layout()
    tampilkan("kmodes_2_ukuran_nilai.png")

    # komposisi fitur pembeda per cluster
    tampil = beda[:6]
    baris = (len(tampil) + 2) // 3
    fig, ax = plt.subplots(baris, 3, figsize=(15, 4 * baris))
    for a, f in zip(np.ravel(ax), tampil):
        pd.crosstab(df["Cluster"], data[f], normalize="index").mul(100).plot(
            kind="bar", stacked=True, ax=a, colormap="coolwarm", rot=0)
        a.set_title(f)
        a.set_ylabel("% siswa")
        a.legend(fontsize=7, loc="upper right")
    for a in np.ravel(ax)[len(tampil):]:
        a.axis("off")
    plt.tight_layout()
    tampilkan("kmodes_3_komposisi.png")

    print("\nHasil disimpan -> output/hasil_clustering_kmodes.csv, centroid_kmodes.csv")


if __name__ == "__main__":
    main()
