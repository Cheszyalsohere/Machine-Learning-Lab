import os
import pandas as pd
from scipy.stats import ttest_ind

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET   = "HeartDisease"
ALPHA    = 0.05

FITUR_NUMERIK = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]


def main():
    df = pd.read_csv(os.path.join(BASE_DIR, "heart.csv"))
    print("=" * 62)
    print("  SELEKSI FITUR - INDEPENDENT t-TEST (fitur numerik)")
    print("=" * 62)
    print(f"Dataset : heart.csv  ({df.shape[0]} baris x {df.shape[1]} kolom)")
    print(f"Target  : {TARGET}  |  alpha = {ALPHA}\n")

    kelas0 = df[df[TARGET] == 0]
    kelas1 = df[df[TARGET] == 1]
    print(f"Jumlah data  ->  kelas 0 (sehat): {len(kelas0)}  |  kelas 1 (sakit): {len(kelas1)}\n")

    hasil = []
    for fitur in FITUR_NUMERIK:
        t_stat, p = ttest_ind(kelas0[fitur], kelas1[fitur], equal_var=False)
        signifikan = p < ALPHA
        hasil.append({
            "Fitur": fitur,
            "Mean_Kelas0": round(kelas0[fitur].mean(), 3),
            "Mean_Kelas1": round(kelas1[fitur].mean(), 3),
            "T_Statistic": round(t_stat, 4),
            "P_Value": p,
            "Keputusan": "Dipilih (signifikan)" if signifikan else "Dibuang (tidak signifikan)",
        })

    res = pd.DataFrame(hasil).sort_values("P_Value").reset_index(drop=True)

    print(f"{'Fitur':<14}{'Mean_0':>10}{'Mean_1':>10}{'t-stat':>10}{'P-Value':>13}   Keputusan")
    print("-" * 78)
    for _, r in res.iterrows():
        print(f"{r['Fitur']:<14}{r['Mean_Kelas0']:>10.2f}{r['Mean_Kelas1']:>10.2f}"
              f"{r['T_Statistic']:>10.2f}{r['P_Value']:>13.3e}   {r['Keputusan']}")

    dipilih = res[res["Keputusan"].str.startswith("Dipilih")]["Fitur"].tolist()
    print("\nFitur numerik terpilih (p < 0.05):")
    print("  " + (", ".join(dipilih) if dipilih else "(tidak ada)"))

    out = os.path.join(BASE_DIR, "hasil_seleksi_ttest.csv")
    res.to_csv(out, index=False)
    print(f"\nHasil disimpan -> hasil_seleksi_ttest.csv")


if __name__ == "__main__":
    main()
