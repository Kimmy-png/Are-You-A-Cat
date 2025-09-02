import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Are You a Cat?",
    page_icon="🐾",
    layout="centered"
)

# --- FUNGSI UNTUK MEMUAT MODEL ---
# @st.cache_resource digunakan agar model hanya di-load sekali
@st.cache_resource
def load_model():
    """Memuat model Keras yang sudah dilatih."""
    try:
        model = tf.keras.models.load_model('model.keras')
        return model
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None

# --- FUNGSI UNTUK PREDIKSI ---
def predict(model, image):
    """Melakukan prediksi pada gambar yang diunggah."""
    # Ukuran gambar harus sama dengan saat training
    IMG_SIZE = (128, 128)
    
    # Pre-process gambar
    img = image.resize(IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Buat batch

    # Lakukan prediksi
    predictions = model.predict(img_array)
    skor = predictions[0]

    # Dapatkan persentase kemiripan dengan kucing
    persentase_kucing = skor[1] * 100
    return persentase_kucing

# --- ANTARMUKA UTAMA APLIKASI ---

st.title("🐱 Apakah Ini Kucing?")
st.write(
    "Unggah gambar untuk melihat seberapa mirip gambar tersebut dengan kucing "
    "menurut model AI yang telah dilatih pada ribuan gambar kucing dan manusia."
)

# Load model
model = load_model()

if model:
    # Komponen untuk upload file
    uploaded_file = st.file_uploader(
        "Pilih sebuah gambar...", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Tampilkan gambar yang diunggah
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar yang diunggah.', use_column_width=True)
        
        # Tombol untuk memicu prediksi
        if st.button('Analisis Gambar Ini'):
            with st.spinner('Model sedang berpikir...'):
                persentase = predict(model, image)
            
            st.subheader("Hasil Analisis:")
            if persentase > 50:
                st.success(f"Gambar ini lebih mirip Kucing!")
            else:
                st.error(f"Gambar ini lebih mirip Manusia.")
            
            st.progress(int(persentase))
            st.metric(label="Tingkat Kemiripan dengan Kucing", value=f"{persentase:.2f}%")
else:
    st.warning("Model tidak dapat dimuat. Aplikasi tidak dapat berjalan.")
