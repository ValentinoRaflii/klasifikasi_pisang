import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
import os
import base64

# ===== 1. LOAD MODEL (Tetap Berfungsi Seperti Semula) =====
@st.cache_resource
def load_models():
    base = "models"
    model_jenis      = joblib.load(os.path.join(base, "model_svm_jenis.pkl"))
    scaler_jenis     = joblib.load(os.path.join(base, "scaler_jenis.pkl"))
    le_jenis         = joblib.load(os.path.join(base, "labelencoder_jenis.pkl"))
    model_kematangan = joblib.load(os.path.join(base, "model_svm_kematangan.pkl"))
    scaler_kematangan= joblib.load(os.path.join(base, "scaler_kematangan.pkl"))
    le_kematangan    = joblib.load(os.path.join(base, "labelencoder_kematangan.pkl"))
    return model_jenis, scaler_jenis, le_jenis, model_kematangan, scaler_kematangan, le_kematangan

model_jenis, scaler_jenis, le_jenis, model_kematangan, scaler_kematangan, le_kematangan = load_models()

# ===== 2. FUNGSI EKSTRAKSI FITUR =====
def segmentasi_dan_kontur(img_array):
    img = cv2.resize(img_array, (300, 300))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturasi = hsv[:, :, 1]
    _, mask = cv2.threshold(saturasi, 30, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    kontur, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(kontur) == 0:
        return None, None
    kontur_terbesar = max(kontur, key=cv2.contourArea)
    return kontur_terbesar, mask

def ekstraksi_fitur_bentuk(img_array):
    kontur, mask = segmentasi_dan_kontur(img_array)
    if kontur is None:
        return None
    area = cv2.contourArea(kontur)
    perimeter = cv2.arcLength(kontur, True)
    x, y, w, h = cv2.boundingRect(kontur)
    aspect_ratio = float(w) / h if h != 0 else 0
    extent = float(area) / (w * h) if (w * h) != 0 else 0
    hull = cv2.convexHull(kontur)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area != 0 else 0
    if len(kontur) >= 5:
        (_, _), (MA, ma), _ = cv2.fitEllipse(kontur)
        eccentricity = np.sqrt(1 - (min(MA, ma) / max(MA, ma)) ** 2) if max(MA, ma) != 0 else 0
    else:
        eccentricity = 0
    moments = cv2.moments(kontur)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    fitur = [area, perimeter, aspect_ratio, extent, solidity, eccentricity]
    fitur.extend(hu_moments_log)
    return fitur

def ekstraksi_fitur_hsv(img_array, bins=16):
    img = cv2.resize(img_array, (300, 300))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256]).flatten()
    hist_h = hist_h / hist_h.sum()
    hist_s = hist_s / hist_s.sum()
    hist_v = hist_v / hist_v.sum()
    return np.concatenate([hist_h, hist_s, hist_v])

def prediksi(img_array):
    fitur_bentuk = ekstraksi_fitur_bentuk(img_array)
    if fitur_bentuk is None:
        return None, None, None
    fitur_bentuk_scaled = scaler_jenis.transform([fitur_bentuk])
    pred_jenis = le_jenis.inverse_transform(model_jenis.predict(fitur_bentuk_scaled))[0]

    fitur_hsv = ekstraksi_fitur_hsv(img_array)
    fitur_hsv_scaled = scaler_kematangan.transform([fitur_hsv])
    pred_kematangan = le_kematangan.inverse_transform(model_kematangan.predict(fitur_hsv_scaled))[0]

    kontur, mask = segmentasi_dan_kontur(img_array)
    img_kontur = cv2.resize(img_array, (300, 300)).copy()
    if kontur is not None:
        cv2.drawContours(img_kontur, [kontur], -1, (0, 255, 0), 2)

    return pred_jenis, pred_kematangan, img_kontur


# ===== 3. UI STREAMLIT (ORIGINAL BACKGROUND IMAGE - NO OVERLAY) =====
st.set_page_config(
    page_title="Banana Classifier Dashboard",
    page_icon="🍌",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Fungsi untuk membaca file PNG lokal
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

# Membaca background_banana.png yang berada di folder project
bg_image_base64 = get_base64_image("background_banana.png")

st.markdown(f"""
    <style>
    /* Menggunakan gambar asli 100% tanpa lapisan warna penutup */
    .stApp {{
        background-image: url("data:image/png;base64,{bg_image_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    
    /* Perubahan Warna Huruf Judul Utama Menjadi Hijau Tua */
    .hero-title {{
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        color: #064E3B !important; 
        background: none !important;
        -webkit-text-fill-color: initial !important;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }}
    
    /* Perubahan Warna Sub-judul Menjadi Hijau Tua Medium */
    .hero-subtitle {{
        font-size: 1.3rem !important;
        color: #047857 !important; 
        margin-top: 5px;
        margin-bottom: 2.5rem;
        font-weight: 600 !important;
    }}

    /* Sub-heading kolom (Input Citra & Hasil Analisis) disesuaikan ke Hijau Tua */
    h3 {{
        color: #064E3B !important;
    }}

    /* Mengubah warna teks placeholder "Belum Ada Citra" menjadi Hijau Tua */
    .placeholder-text h4 {{
        color: #064E3B !important;
        font-weight: 700 !important;
    }}
    .placeholder-text p {{
        color: #047857 !important;
        font-weight: 500 !important;
    }}
    
    /* Mengubah warna caption ilustrasi gambar menjadi hijau tua */
    div[data-testid="stImageCaption"] {{
        color: #047857 !important;
        font-weight: 600 !important;
    }}

    /* Box Konten dibuat gelap solid agar isi teks/grafik di dalamnya tidak bertabrakan dengan background */
    div[data-testid="stDataframeCard"] {{
        background: rgba(15, 23, 42, 0.93) !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 16px !important;
    }}
    
    .stElementContainer div[data-testid="element-container"] .stVerticalBlockBorderWrapper {{
        background: rgba(15, 23, 42, 0.93) !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 14px !important;
        padding: 1.5rem !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }}
    </style>
""", unsafe_allow_html=True)

# ===== SIDEBAR: INFORMASI & DAFTAR KELAS =====
with st.sidebar:
    st.image("https://img.icons8.com/plasticine/100/banana.png", width=80)
    st.markdown("## **Info & Konfigurasi**")
    st.markdown("Detail model *Support Vector Machine* (SVM) yang berjalan di latar belakang:")
    
    with st.container(border=True):
        st.markdown("🔹 **Model Jenis Pisang**")
        st.caption(f"Jumlah Kelas: {len(le_jenis.classes_)}")
        st.caption("Kernel: RBF | Akurasi: **96.40%**")
        
    with st.container(border=True):
        st.markdown("🔸 **Model Kematangan**")
        st.caption(f"Jumlah Kelas: {len(le_kematangan.classes_)}")
        st.caption("Kernel: Linear | Akurasi: **99.33%**")

    st.markdown("---")
    st.markdown("### 📋 Target Kelas")
    
    col_list1, col_list2 = st.columns(2)
    with col_list1:
        st.markdown("**Jenis:**")
        for kelas in le_jenis.classes_:
            st.markdown(f"• `{kelas}`")
    with col_list2:
        st.markdown("**Kematangan:**")
        for kelas in le_kematangan.classes_:
            st.markdown(f"• `{kelas}`")
            
    st.markdown("---")
    st.caption("🤖 Kelompok 8 | Teknik Informatika | Universitas Halu Oleo")


# ===== HALAMAN UTAMA =====
st.markdown('<h1 class="hero-title">🍌 Banana Inteligência</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Sistem Deteksi Jenis dan Tingkat Kematangan Pisang Berbasis Machine Learning (SVM)</p>', unsafe_allow_html=True)

# Grid Layout: Kiri (Upload), Kanan (Hasil / Placeholder)
col_left, col_right = st.columns([4, 6], gap="large")

with col_left:
    st.markdown("### 📥 Input Citra")
    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "Seret atau pilih gambar pisang untuk dianalisis",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💡 **Petunjuk Kesuksesan Deteksi:**\n\nPastikan foto pisang diambil dengan pencahayaan yang cukup dan menggunakan latar belakang polos yang kontras agar pemisahan objek (segmentasi kontur) berhasil secara presisi.")
        
        # ===== MEMBACA GAMBAR CONTOH LOKAL MANDIRI =====
        st.markdown("<br>", unsafe_allow_html=True)
        nama_gambar_contoh = "contoh_dataset.jpg" # Ganti nama file ini sesuai file gambarmu sendiri
        
        if os.path.exists(nama_gambar_contoh):
            img_contoh = Image.open(nama_gambar_contoh)
            st.image(
                img_contoh, 
                caption="Contoh Citra Dataset yang Benar (Satu Objek Jelas, Background Kontras)",
                use_container_width=True
            )
        else:
            st.warning(f"Silakan simpan file gambar contoh dengan nama '{nama_gambar_contoh}' di folder project agar tampil di sini.")

with col_right:
    st.markdown("### 🔍 Hasil Analisis")
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        with st.spinner("Mengekstrak fitur bentuk & warna..."):
            pred_jenis, pred_kematangan, img_kontur = prediksi(img_bgr)

        if pred_jenis is None:
            st.error("⚠️ **Gagal Mendeteksi Objek** — Sistem tidak berhasil menemukan kontur pisang yang jelas. Silakan ganti foto.")
        else:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.subheader("🍌 Jenis Pisang")
                st.info(f"### **{pred_jenis}**")
            with col_m2:
                st.subheader("🟢 Kematangan")
                if "mentah" in pred_kematangan.lower():
                    st.warning(f"### **{pred_kematangan}**")
                elif "masak" in pred_kematangan.lower() or "matang" in pred_kematangan.lower():
                    st.success(f"### **{pred_kematangan}**")
                else:
                    st.error(f"### **{pred_kematangan}**")

            st.markdown("<br>", unsafe_allow_html=True)

            tab1, tab2 = st.tabs(["🖼️ Gambar Asli", "🎯 Hasil Segmentasi & Kontur"])
            with tab1:
                st.image(img_rgb, use_container_width=True)
            with tab2:
                img_kontur_rgb = cv2.cvtColor(img_kontur, cv2.COLOR_BGR2RGB)
                st.image(img_kontur_rgb, use_container_width=True)
                
    else:
        with st.container(border=True):
            st.markdown(
                "<div class='placeholder-text' style='text-align: center; padding: 1rem 0rem;'>"
                "<h4>Belum Ada Citra yang Diunggah</h4>"
                "<p>Silakan unggah berkas foto di panel sebelah kiri untuk memulai analisis otomatis.</p>"
                "</div>", 
                unsafe_allow_html=True
            )
            
            st.image(
                "https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?q=80&w=800&auto=format&fit=crop", 
                caption="Ilustrasi: Pemindaian Pisang berbasis Computer Vision",
                use_container_width=True
            )