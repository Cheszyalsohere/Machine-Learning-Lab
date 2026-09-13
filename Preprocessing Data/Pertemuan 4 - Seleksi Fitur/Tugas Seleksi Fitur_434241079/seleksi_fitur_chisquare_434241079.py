# =============================================================
#  TUGAS SELEKSI FITUR - MACHINE LEARNING (Pertemuan 4)
#  File   : seleksi_fitur_chisquare_434241079.py
#  NIM    : 434241079
#  Metode : Chi-Square  -> untuk FITUR KATEGORIKAL
#  Dataset: heart.csv   (target = HeartDisease, 0/1)
#
#  Ide: uji apakah tiap fitur kategorikal punya hubungan
#       signifikan dengan target. Bila p-value < 0.05 -> fitur
#       dianggap RELEVAN dan layak dipilih.
# =============================================================

import os
import pandas as pd
from scipy.stats import chi2_contingency

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET   = "HeartDisease"
ALPHA    = 0.05

# Fitur kategorikal (bertipe teks) + FastingBS yang bernilai biner 0/1
FITUR_KATEGORIK = ["Sex", "ChestPainType", "RestingECG",
                   "ExerciseAngina", "ST_Slope", "FastingBS"]


def main():
    df = pd.read_csv(os.path.join(BASE_DIR, "heart.csv"))
    print("=" * 62)
    print("  SELEKSI FITUR - CHI-SQUARE (fitur kategorikal)")
    print("=" * 62)
    print(f"Dataset : heart.csv  ({df.shape[0]} baris x {df.shape[1]} kolom)")
    print(f"Target  : {TARGET}  |  alpha = {ALPHA}\n")

    hasil = []
    for fitur in FITUR_KATEGORIK:
        # tabel kontingensi antara fitur dan target
        tabel = pd.crosstab(df[fitur], df[TARGET])
        chi2, p, dof, _ = chi2_contingency(tabel)
        signifikan = p < ALPHA
        hasil.append({
            "Fitur": fitur,
            "Chi_Square": round(chi2, 4),
            "P_Value": p,
            "DOF": dof,
            "Keputusan": "Dipilih (signifikan)" if signifikan else "Dibuang (tidak signifikan)",
        })

    # urutkan dari yang paling signifikan (p terkecil)
    res = pd.DataFrame(hasil).sort_values("P_Value").reset_index(drop=True)

    # --- tampilkan tabel hasil dengan rapi ---
    print(f"{'Fitur':<16}{'Chi-Square':>12}{'P-Value':>14}   Keputusan")
    print("-" * 62)
    for _, r in res.iterrows():
        print(f"{r['Fitur']:<16}{r['Chi_Square']:>12.2f}{r['P_Value']:>14.3e}   {r['Keputusan']}")

    dipilih = res[res["Keputusan"].str.startswith("Dipilih")]["Fitur"].tolist()
    print("\nFitur kategorikal terpilih (p < 0.05):")
    print("  " + (", ".join(dipilih) if dipilih else "(tidak ada)"))

    # --- simpan hasil ke CSV ---
    out = os.path.join(BASE_DIR, "hasil_seleksi_chisquare.csv")
    res.to_csv(out, index=False)
    print(f"\nHasil disimpan -> hasil_seleksi_chisquare.csv")


if __name__ == "__main__":
    main()
