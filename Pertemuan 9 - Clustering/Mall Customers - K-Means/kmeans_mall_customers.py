# =============================================================
#  TUGAS CLUSTERING - MACHINE LEARNING (Pertemuan 9)
#  File    : kmeans_mall_customers.py
#  Kelompok: 5 (TI-C5) - Muhammad Irfan Nuha (434241079)
#                        Bunga Lala Sal Sabila (434241066)
#  Isi     : Melengkapi kode modul (bagian titik-titik) K-Means
#            pada dataset Mall_Customers.csv
# =============================================================

import os
import warnings
warnings.filterwarnings("ignore")

# 1
# import tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)


def tampilkan(nama_file):
    """Simpan figure ke folder output lalu tampilkan (plt.show)."""
    plt.savefig(os.path.join(OUT_DIR, nama_file), dpi=110, bbox_inches="tight")
    plt.show()
    plt.close()


def judul(teks):
    print("\n" + "=" * 62)
    print(f"  {teks}")
    print("=" * 62)


# 2
# import data
df = pd.read_csv(os.path.join(BASE_DIR, "Mall_Customers.csv"))
judul("2. IMPORT DATA - df.head()")
print(df.head())

# 3
# amati bentuk data
judul("3. BENTUK DATA - df.shape")
print(df.shape)

# 4
# Melihat ringkasan statistik deskriptif dari DataFrame
judul("4. STATISTIK DESKRIPTIF - df.describe()")
print(df.describe().round(2))

# 5
# cek null data
judul("5. CEK NULL - df.isnull().sum()")
print(df.isnull().sum())

# 6
# cek outlier
judul("6. CEK OUTLIER (metode IQR)")
kolom_numerik = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
for kol in kolom_numerik:
    q1, q3 = df[kol].quantile(0.25), df[kol].quantile(0.75)
    iqr = q3 - q1
    bawah, atas = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    jumlah = ((df[kol] < bawah) | (df[kol] > atas)).sum()
    print(f"   {kol:<24} batas [{bawah:7.2f}, {atas:7.2f}]  outlier = {jumlah}")
plt.figure(figsize=(12, 4))
for i, kol in enumerate(kolom_numerik, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(y=df[kol], color="#8fb3e0")
    plt.title(kol)
plt.tight_layout()
tampilkan("mall_6_boxplot_outlier.png")

# 7
# amati bentuk visual masing-masing fitur
plt.style.use('fivethirtyeight')
plt.figure(1, figsize=(15, 6))
n = 0
for x in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1, 3, n)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    sns.histplot(
        df[x], kde=True,
        stat="density", kde_kws=dict(cut=3), bins=20)
    plt.title('Distplot of {}'.format(x))
tampilkan("mall_7_distribusi.png")

# 8
# Ploting untuk mencari relasi antara Age , Annual Income and Spending Score
plt.figure(1, figsize=(15, 20))
n = 0
for x in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    for y in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
        n += 1
        plt.subplot(3, 3, n)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        sns.regplot(x=x, y=y, data=df)
        plt.ylabel(y.split()[0] + ' ' + y.split()[1] if len(y.split()) > 1 else y)
tampilkan("mall_8_relasi.png")

# 9
# Melihat sebaran Spending Score dan Annual Income pada Gender
plt.figure(1, figsize=(15, 8))
for gender in ['Male', 'Female']:
    plt.scatter(x='Annual Income (k$)', y='Spending Score (1-100)',
                data=df[df['Gender'] == gender], s=200, alpha=0.5,
                label=gender)
    plt.xlabel('Annual Income (k$)'), plt.ylabel('Spending Score (1-100)')
    plt.title('Annual Income vs Spending Score')
    plt.legend()
tampilkan("mall_9_gender.png")

# 10
# Merancang K-Means untuk spending score vs annual income
# Menentukan nilai k yang sesuai dengan Elbow-Method
X1 = df[['Annual Income (k$)', 'Spending Score (1-100)']].iloc[:, :].values
inertia = []
for n in range(1, 11):
    algorithm = (KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300, random_state=111))
    algorithm.fit(X1)
    inertia.append(algorithm.inertia_)

judul("10. ELBOW METHOD - nilai inertia")
for k, nilai in enumerate(inertia, 1):
    print(f"   k = {k:<2}  inertia = {nilai:12.2f}")

# Plot bentuk visual elbow
plt.figure(1, figsize=(15, 6))
plt.plot(range(1, 11), inertia, 'o')
plt.plot(range(1, 11), inertia, '-', alpha=0.5)
plt.xlabel('Number of Clusters'), plt.ylabel('Inertia')
tampilkan("mall_10_elbow.png")

# 11
# Membangun K-Means
algorithm = (KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300,
                    tol=0.0001, random_state=111, algorithm='elkan'))
algorithm.fit(X1)
labels2 = algorithm.labels_
centroids2 = algorithm.cluster_centers_

judul("11. K-MEANS (k = 5) - centroid tiap cluster")
print(pd.DataFrame(centroids2, columns=["Annual Income (k$)", "Spending Score (1-100)"]).round(2))
print("\nJumlah anggota tiap cluster:")
print(pd.Series(labels2).value_counts().sort_index().to_string())

# 12
# Menyiapkan data untuk bentuk visual cluster
labels2 = algorithm.labels_
centroids2 = algorithm.cluster_centers_
step = 0.02
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
Z1 = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])  # array diratakan 1D

# 13
# Melihat bentuk visual cluster
plt.figure(1, figsize=(15, 7))
plt.clf()
Z1 = Z1.reshape(xx.shape)
plt.imshow(Z1, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Pastel2, aspect='auto', origin='lower')
plt.scatter(x='Annual Income (k$)', y='Spending Score (1-100)', data=df, c=labels2, s=200)
plt.scatter(x=centroids2[:, 0], y=centroids2[:, 1], s=300, c='red', alpha=0.5)
plt.ylabel('Spending Score (1-100)'), plt.xlabel('Annual Income (k$)')
tampilkan("mall_13_cluster.png")

# 14
# Melihat nilai Silhouette Score
judul("14. SILHOUETTE SCORE")
score2 = silhouette_score(X1, labels2)
print("Silhouette Score: ", score2)
