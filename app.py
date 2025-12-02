import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Sistem Peringatan Dini Retensi Mahasiswa",
    page_icon="🎓",
    layout="wide"
)

# Judul dan Deskripsi
st.title("🎓 Sistem Peringatan Dini Retensi Mahasiswa")
st.markdown("""
Aplikasi ini menggunakan algoritma **Random Forest** untuk memprediksi potensi mahasiswa 
apakah akan **Dropout**, **Enrolled** (Masih Kuliah), atau **Graduate** (Lulus).
Dibuat untuk memenuhi tugas Project Kelompok.
""")

# --- 1. LOAD & PREPROCESS DATA ---
@st.cache_data
def load_data():
    # Membaca dataset
    df = pd.read_csv('dataset.csv')
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("File 'dataset.csv' tidak ditemukan. Pastikan file csv ada di folder yang sama.")
    st.stop()

# --- 2. FITUR SELEKSI & PERSIAPAN MODEL ---
# Kita akan menggunakan fitur-fitur yang paling berpengaruh (Top Features) 
# agar input user tidak terlalu banyak (30+ input membingungkan user).
# Fitur dipilih berdasarkan korelasi umum pada dataset ini.

selected_features = [
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Tuition fees up to date',
    'Scholarship holder',
    'Debtor',
    'Age at enrollment',
    'Gender',
    'Displaced'
]

target_col = 'Target'

# Filter dataframe hanya kolom yang dipilih + target
df_model = df[selected_features + [target_col]].copy()

# Encoding Target (Dropout, Enrolled, Graduate) menjadi angka jika belum
le = LabelEncoder()
df_model[target_col] = le.fit_transform(df_model[target_col])

# Mapping untuk menampilkan hasil nanti
# Biasanya urutan le.classes_ alfabetis: Dropout, Enrolled, Graduate
target_names = le.classes_

# Pisahkan X dan y
X = df_model[selected_features]
y = df_model[target_col]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. TRAINING MODEL RANDOM FOREST ---
@st.cache_resource
def train_model(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf

rf_model = train_model(X_train, y_train)

# Hitung Akurasi
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# --- 4. SIDEBAR INPUT USER ---
st.sidebar.header("📝 Input Data Mahasiswa")

def user_input_features():
    # Kelompok Akademik
    st.sidebar.subheader("Akademik")
    sem2_approved = st.sidebar.number_input("SKS Lulus (Sem 2)", min_value=0, max_value=30, value=5)
    sem2_grade = st.sidebar.number_input("Nilai Rata-rata (Sem 2)", min_value=0.0, max_value=20.0, value=12.0)
    sem1_approved = st.sidebar.number_input("SKS Lulus (Sem 1)", min_value=0, max_value=30, value=5)
    sem1_grade = st.sidebar.number_input("Nilai Rata-rata (Sem 1)", min_value=0.0, max_value=20.0, value=12.0)
    
    # Kelompok Finansial
    st.sidebar.subheader("Finansial")
    tuition = st.sidebar.selectbox("Uang Kuliah Lancar?", [1, 0], format_func=lambda x: "Ya" if x==1 else "Tidak")
    scholarship = st.sidebar.selectbox("Penerima Beasiswa?", [1, 0], format_func=lambda x: "Ya" if x==1 else "Tidak")
    debtor = st.sidebar.selectbox("Memiliki Utang?", [1, 0], format_func=lambda x: "Ya" if x==1 else "Tidak")
    
    # Kelompok Demografi
    st.sidebar.subheader("Demografi")
    age = st.sidebar.slider("Usia saat Mendaftar", 17, 70, 20)
    gender = st.sidebar.selectbox("Gender", [1, 0], format_func=lambda x: "Laki-laki" if x==1 else "Perempuan")
    displaced = st.sidebar.selectbox("Perantau (Displaced)?", [1, 0], format_func=lambda x: "Ya" if x==1 else "Tidak")

    data = {
        'Curricular units 2nd sem (approved)': sem2_approved,
        'Curricular units 2nd sem (grade)': sem2_grade,
        'Curricular units 1st sem (approved)': sem1_approved,
        'Curricular units 1st sem (grade)': sem1_grade,
        'Tuition fees up to date': tuition,
        'Scholarship holder': scholarship,
        'Debtor': debtor,
        'Age at enrollment': age,
        'Gender': gender,
        'Displaced': displaced
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- 5. MAIN PAGE DISPLAY ---

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Hasil Prediksi")
    
    # Tampilkan input user (opsional)
    with st.expander("Lihat Data Input"):
        st.dataframe(input_df)

    if st.button("Jalankan Prediksi"):
        prediction = rf_model.predict(input_df)
        prediction_proba = rf_model.predict_proba(input_df)
        
        result = target_names[prediction][0]
        
        st.markdown("---")
        if result == "Dropout":
            st.error(f"### ⚠️ Peringatan: Mahasiswa Berisiko DROPOUT")
            st.write("Sistem mendeteksi pola yang mengarah pada pemberhentian studi. Disarankan untuk memberikan konseling akademik segera.")
        elif result == "Enrolled":
            st.info(f"### ℹ️ Status: ENROLLED (Aktif)")
            st.write("Mahasiswa masih dalam jalur pendidikan, namun perlu pemantauan berkala.")
        else:
            st.success(f"### ✅ Status: Lulus (GRADUATE)")
            st.write("Mahasiswa memiliki performa yang baik dan diprediksi akan lulus.")
            
        st.markdown("---")
        st.write("Probabilitas Prediksi:")
        prob_df = pd.DataFrame(prediction_proba, columns=target_names)
        st.bar_chart(prob_df.T)

with col2:
    st.subheader("📊 Performa Model")
    st.metric(label="Akurasi Model", value=f"{accuracy:.2%}")
    st.write("Fitur Terpenting:")
    
    # Feature Importance Visualization
    importances = rf_model.feature_importances_
    feature_imp = pd.DataFrame({'Fitur': selected_features, 'Pentingnya': importances})
    feature_imp = feature_imp.sort_values(by='Pentingnya', ascending=False).head(5)
    st.dataframe(feature_imp, hide_index=True)

# Footer
st.markdown("---")
st.caption("Dikembangkan menggunakan Python & Streamlit | Algoritma Random Forest")
