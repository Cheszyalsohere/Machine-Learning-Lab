# =============================================================
#  UI VERSI 2 - PREDIKSI KEPUASAN PENUMPANG (Streamlit)
#  File    : app_streamlit_ori_434241079.py
#  Kelompok: 5 (TI-C5) - Muhammad Irfan Nuha (434241079)
#                        Bunga Lala Sal Sabila (434241066)
#  Model   : Decision Tree (Ori) - tanpa transformasi & seleksi fitur,
#            akurasi tertinggi dari semua metode yang dicoba
#
#  Cara menjalankan:
#    1. python klasifikasi_434241079.py          (melatih & menyimpan model)
#    2. streamlit run app_streamlit_ori_434241079.py
# =============================================================

import os
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_MODEL = os.path.join(BASE_DIR, "model_terbaik_ori.joblib")

LABEL_FITUR = {
    "Gender": "Jenis Kelamin",
    "Customer Type": "Tipe Pelanggan",
    "Age": "Usia",
    "Type of Travel": "Tujuan Perjalanan",
    "Class": "Kelas Penerbangan",
    "Flight Distance": "Jarak Penerbangan (mil)",
    "Inflight wifi service": "Wi-Fi di Pesawat",
    "Departure/Arrival time convenient": "Ketepatan Jadwal",
    "Ease of Online booking": "Kemudahan Booking Online",
    "Gate location": "Lokasi Gate",
    "Food and drink": "Makanan & Minuman",
    "Online boarding": "Online Boarding",
    "Seat comfort": "Kenyamanan Kursi",
    "Inflight entertainment": "Hiburan di Pesawat",
    "On-board service": "Pelayanan Awak Kabin",
    "Leg room service": "Ruang Kaki",
    "Baggage handling": "Penanganan Bagasi",
    "Checkin service": "Layanan Check-in",
    "Inflight service": "Layanan Selama Penerbangan",
    "Cleanliness": "Kebersihan",
    "Departure Delay in Minutes": "Keterlambatan Berangkat (menit)",
    "Arrival Delay in Minutes": "Keterlambatan Tiba (menit)",
}
LABEL_NILAI = {
    "Loyal Customer": "Pelanggan setia",
    "disloyal Customer": "Pelanggan tidak setia",
    "Business travel": "Bisnis",
    "Personal Travel": "Pribadi",
    "Business": "Business",
    "Eco": "Economy",
    "Eco Plus": "Economy Plus",
    "Female": "Perempuan",
    "Male": "Laki-laki",
}
CONTOH = {
    "puas": {"kategori": {"Gender": "Female", "Customer Type": "Loyal Customer",
                          "Type of Travel": "Business travel", "Class": "Business"},
             "rating": 5,
             "angka": {"Age": 40, "Flight Distance": 2500,
                       "Departure Delay in Minutes": 0, "Arrival Delay in Minutes": 0}},
    "tidak": {"kategori": {"Gender": "Male", "Customer Type": "disloyal Customer",
                           "Type of Travel": "Personal Travel", "Class": "Eco"},
              "rating": 2,
              "angka": {"Age": 25, "Flight Distance": 500,
                        "Departure Delay in Minutes": 25, "Arrival Delay in Minutes": 25}},
}


@st.cache_resource
def muat_model():
    return joblib.load(PATH_MODEL)


def jenis(bundle, f):
    """Kelompokkan fitur: kategori, rating (0-5), atau angka."""
    if f in bundle["encoder"]:
        return "kategori"
    return "rating" if bundle["rentang"][f][1] <= 5 else "angka"


def isi_contoh(bundle, kunci):
    """Isi semua input dengan nilai contoh (disimpan di session_state)."""
    c = CONTOH[kunci]
    for f in bundle["fitur_terpilih"]:
        j = jenis(bundle, f)
        if j == "kategori":
            st.session_state[f] = c["kategori"].get(f, bundle["encoder"][f][0])
        elif j == "rating":
            st.session_state[f] = c["rating"]
        else:
            lo, hi = bundle["rentang"][f]
            st.session_state[f] = float(min(max(c["angka"].get(f, bundle["default"][f]), lo), hi))


def prediksi(bundle, masukan):
    """Encoding -> capping outlier -> Decision Tree (tanpa Min-Max & seleksi fitur)."""
    baris = dict(bundle["default"])
    for f, nilai in masukan.items():
        if f in bundle["encoder"]:
            baris[f] = bundle["encoder"][f].index(nilai)
        else:
            lo, hi = bundle["batas_outlier"].get(f, (float("-inf"), float("inf")))
            baris[f] = min(max(float(nilai), lo), hi)
    data = pd.DataFrame([baris])[bundle["urutan_fitur"]]
    model = bundle["model"]
    kelas = int(model.predict(data)[0])
    jumlah_aturan = int(model.decision_path(data).indices.size - 1)
    return bundle["label"][kelas], jumlah_aturan


# ======================= TAMPILAN =======================
st.set_page_config(page_title="Prediksi Kepuasan Penumpang (Versi 2)",
                   page_icon=":deciduous_tree:", layout="wide")

if not os.path.exists(PATH_MODEL):
    st.error("File model_terbaik_ori.joblib belum ada. Jalankan dulu: python klasifikasi_434241079.py")
    st.stop()

bundle = muat_model()
fitur = bundle["fitur_terpilih"]

contoh_url = st.query_params.get("contoh")
if contoh_url in CONTOH and st.session_state.get("_contoh") != contoh_url:
    isi_contoh(bundle, contoh_url)
    st.session_state["_contoh"] = contoh_url

st.title(":airplane: Prediksi Kepuasan Penumpang Maskapai")
st.caption(f"UI Versi 2 · Model: **{bundle['nama_model']}** (tanpa transformasi & seleksi fitur) · "
           f"akurasi data testing **{bundle['akurasi'] * 100:.2f}%** · Kelompok 5 TI-C5")

with st.sidebar:
    st.header("Tentang Model")
    st.write(f"Model terpilih: **{bundle['nama_model']}**")
    st.metric("Akurasi (data testing)", f"{bundle['akurasi'] * 100:.2f}%")
    st.write("Perbandingan semua metode:")
    st.dataframe(pd.DataFrame(bundle["perbandingan"])[["Model", "Akurasi"]],
                 hide_index=True, width="stretch")
    st.write("5 fitur paling berpengaruh:")
    penting = pd.Series(bundle["model"].feature_importances_, index=bundle["urutan_fitur"])
    penting = penting.sort_values(ascending=False).head(5)
    st.dataframe(pd.DataFrame({"Fitur": [LABEL_FITUR.get(f, f) for f in penting.index],
                               "Pengaruh": (penting.values * 100).round(1)}),
                 hide_index=True, width="stretch")
    st.divider()
    st.write("Isi cepat dengan contoh:")
    if st.button("Penumpang puas", icon=":material/sentiment_satisfied:", width="stretch"):
        isi_contoh(bundle, "puas")
    if st.button("Penumpang tidak puas", icon=":material/sentiment_dissatisfied:",
                 width="stretch"):
        isi_contoh(bundle, "tidak")

masukan = {}
kol_profil, kol_layanan = st.columns([1, 2], gap="large")

with kol_profil:
    st.subheader(":briefcase: Profil Perjalanan")
    for f in fitur:
        j = jenis(bundle, f)
        if j == "kategori":
            pilihan = bundle["encoder"][f]
            if f not in st.session_state:
                st.session_state[f] = pilihan[0]
            masukan[f] = st.selectbox(LABEL_FITUR.get(f, f), pilihan, key=f,
                                      format_func=lambda v: LABEL_NILAI.get(v, v))
        elif j == "angka":
            lo, hi = bundle["rentang"][f]
            if f not in st.session_state:
                st.session_state[f] = float(bundle["default"][f])
            langkah = 50.0 if f == "Flight Distance" else 1.0
            masukan[f] = st.number_input(LABEL_FITUR.get(f, f), min_value=float(lo),
                                         max_value=float(hi), step=langkah, key=f)

with kol_layanan:
    st.subheader(":star: Penilaian Layanan")
    st.caption("Skala 0–5 · 0 = tidak dinilai, 1 = sangat buruk, 5 = sangat baik")
    rating = [f for f in fitur if jenis(bundle, f) == "rating"]
    sub_kiri, sub_kanan = st.columns(2)
    for i, f in enumerate(rating):
        if f not in st.session_state:
            st.session_state[f] = 3
        with (sub_kiri if i % 2 == 0 else sub_kanan):
            masukan[f] = st.slider(LABEL_FITUR.get(f, f), 0, 5, key=f)

st.divider()
hasil, jumlah_aturan = prediksi(bundle, masukan)
with st.container(border=True):
    st.subheader("Hasil Prediksi")
    if hasil == "satisfied":
        st.success("### PUAS (satisfied)\nPenumpang ini diprediksi **puas** dengan penerbangannya.",
                   icon=":material/sentiment_satisfied:")
    else:
        st.error("### NETRAL / TIDAK PUAS\nPenumpang ini diprediksi **netral atau tidak puas**.",
                 icon=":material/sentiment_dissatisfied:")
    st.caption(f"Decision Tree mengambil keputusan setelah menelusuri **{jumlah_aturan} aturan if-else** "
               "dari akar sampai daun pohon.")
