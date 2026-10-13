# =============================================================
#  UI PREDIKSI KEPUASAN PENUMPANG (Streamlit)
#  File    : app_streamlit_434241079.py
#  Kelompok: 5 (TI-C5) - Muhammad Irfan Nuha (434241079)
#                        Bunga Lala Sal Sabila (434241066)
#  Model   : model dengan akurasi tertinggi dari klasifikasi_434241079.py
#
#  Cara menjalankan:
#    1. python klasifikasi_434241079.py      (melatih & menyimpan model)
#    2. streamlit run app_streamlit_434241079.py
# =============================================================

import os
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_MODEL = os.path.join(BASE_DIR, "model_terbaik.joblib")

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
    "puas": {"kategori": {"Customer Type": "Loyal Customer", "Type of Travel": "Business travel",
                          "Class": "Business", "Gender": "Female"},
             "rating": 5, "angka": {"Flight Distance": 2500, "Age": 40}},
    "tidak": {"kategori": {"Customer Type": "disloyal Customer", "Type of Travel": "Personal Travel",
                           "Class": "Eco", "Gender": "Male"},
              "rating": 2, "angka": {"Flight Distance": 500, "Age": 25}},
}


@st.cache_resource
def muat_model():
    return joblib.load(PATH_MODEL)


def isi_contoh(bundle, kunci):
    """Isi semua input dengan nilai contoh (disimpan di session_state)."""
    c = CONTOH[kunci]
    for f in bundle["fitur_terpilih"]:
        if f in bundle["encoder"]:
            st.session_state[f] = c["kategori"].get(f, bundle["encoder"][f][0])
        elif bundle["rentang"][f][1] <= 5:
            st.session_state[f] = c["rating"]
        else:
            lo, hi = bundle["rentang"][f]
            st.session_state[f] = float(min(max(c["angka"].get(f, (lo + hi) / 2), lo), hi))


def prediksi(bundle, masukan):
    """Ubah input menjadi baris data, lalu: capping -> Min-Max -> seleksi fitur -> model."""
    baris = dict(bundle["default"])
    for f, nilai in masukan.items():
        if f in bundle["encoder"]:
            baris[f] = bundle["encoder"][f].index(nilai)
        else:
            lo, hi = bundle["batas_outlier"].get(f, (float("-inf"), float("inf")))
            baris[f] = min(max(float(nilai), lo), hi)
    data = pd.DataFrame([baris])[bundle["urutan_fitur"]]
    data = pd.DataFrame(bundle["scaler"].transform(data), columns=bundle["urutan_fitur"])
    data = bundle["selector"].transform(data)
    kelas = int(bundle["model"].predict(data)[0])
    skor = float(bundle["model"].decision_function(data)[0]) \
        if hasattr(bundle["model"], "decision_function") else None
    return bundle["label"][kelas], skor


# ======================= TAMPILAN =======================
st.set_page_config(page_title="Prediksi Kepuasan Penumpang", page_icon=":airplane:", layout="wide")

if not os.path.exists(PATH_MODEL):
    st.error("File model_terbaik.joblib belum ada. Jalankan dulu: python klasifikasi_434241079.py")
    st.stop()

bundle = muat_model()
fitur = bundle["fitur_terpilih"]

contoh_url = st.query_params.get("contoh")
if contoh_url in CONTOH and st.session_state.get("_contoh") != contoh_url:
    isi_contoh(bundle, contoh_url)
    st.session_state["_contoh"] = contoh_url

st.title(":airplane: Prediksi Kepuasan Penumpang Maskapai")
st.caption(f"Model: **{bundle['nama_model']}** · akurasi data testing "
           f"**{bundle['akurasi'] * 100:.2f}%** · Kelompok 5 TI-C5")

with st.sidebar:
    st.header("Tentang Model")
    st.write(f"Model terpilih: **{bundle['nama_model']}**")
    st.metric("Akurasi (data testing)", f"{bundle['akurasi'] * 100:.2f}%")
    st.write("Perbandingan dengan metode lain:")
    st.dataframe(pd.DataFrame(bundle["perbandingan"])[["Model", "Akurasi"]],
                 hide_index=True, width="stretch")
    st.divider()
    st.write("Isi cepat dengan contoh:")
    if st.button("Penumpang puas", icon=":material/sentiment_satisfied:", width="stretch"):
        isi_contoh(bundle, "puas")
    if st.button("Penumpang tidak puas", icon=":material/sentiment_dissatisfied:",
                 width="stretch"):
        isi_contoh(bundle, "tidak")

kol_kiri, kol_kanan = st.columns(2, gap="large")
masukan = {}

with kol_kiri:
    st.subheader(":briefcase: Profil Perjalanan")
    for f in fitur:
        if f in bundle["encoder"]:
            pilihan = bundle["encoder"][f]
            if f not in st.session_state:
                st.session_state[f] = pilihan[0]
            masukan[f] = st.selectbox(LABEL_FITUR.get(f, f), pilihan, key=f,
                                      format_func=lambda v: LABEL_NILAI.get(v, v))
        elif bundle["rentang"][f][1] > 5:
            lo, hi = bundle["rentang"][f]
            if f not in st.session_state:
                st.session_state[f] = float(bundle["default"][f])
            masukan[f] = st.number_input(LABEL_FITUR.get(f, f), min_value=float(lo),
                                         max_value=float(hi), step=50.0, key=f)

with kol_kanan:
    st.subheader(":star: Penilaian Layanan")
    st.caption("Skala 0–5 · 0 = tidak dinilai, 1 = sangat buruk, 5 = sangat baik")
    for f in fitur:
        if f not in bundle["encoder"] and bundle["rentang"][f][1] <= 5:
            if f not in st.session_state:
                st.session_state[f] = 3
            masukan[f] = st.slider(LABEL_FITUR.get(f, f), 0, 5, key=f)

st.divider()
hasil, skor = prediksi(bundle, masukan)
puas = hasil == "satisfied"
with st.container(border=True):
    st.subheader("Hasil Prediksi")
    if puas:
        st.success("### PUAS (satisfied)\nPenumpang ini diprediksi **puas** dengan penerbangannya.",
                   icon=":material/sentiment_satisfied:")
    else:
        st.error("### NETRAL / TIDAK PUAS\nPenumpang ini diprediksi **netral atau tidak puas**.",
                 icon=":material/sentiment_dissatisfied:")
    if skor is not None:
        st.caption(f"Skor keputusan SVM: {skor:+.2f} — positif cenderung *puas*, negatif cenderung "
                   "*tidak puas*; makin jauh dari 0 makin yakin.")
