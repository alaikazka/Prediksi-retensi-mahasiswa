# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Early Warning Retensi Mahasiswa", layout="wide")

RISK_THRESH_DEFAULT = 0.60
MID_THRESH_DEFAULT = 0.40

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace("\\", "").replace("  ", " ").strip() for c in df.columns]
    # Normalisasi nama umum
    if "Course \\" in df.columns:
        df = df.rename(columns={"Course \\": "Course"})
    if "Target " in df.columns and "Target" not in df.columns:
        df = df.rename(columns={"Target ": "Target"})
    if " Curricular units 2nd sem (grade) " in df.columns:
        df = df.rename(columns={" Curricular units 2nd sem (grade) ": "Curricular units 2nd sem (grade)"})
    if "Curricular units 1st sem (grade) " in df.columns:
        df = df.rename(columns={"Curricular units 1st sem (grade) ": "Curricular units 1st sem (grade)"})
    return df

@st.cache_resource
def load_artifacts():
    model = joblib.load("artifacts/retention_model.joblib")
    label_enc = joblib.load("artifacts/label_encoder.joblib")
    try:
        feat_imp = pd.read_csv("artifacts/feature_importance.csv")
    except:
        feat_imp = None
    return model, label_enc, feat_imp

def infer_and_flag(model, df, risk_thresh, mid_thresh):
    # Pastikan kolom input sesuai dengan yang dipakai saat training
    model_features = model.named_steps["model"].base_estimator.feature_names_in_ \
        if hasattr(model.named_steps["model"].base_estimator, "feature_names_in_") else df.columns
    # Reorder/align columns: jika ada mismatch, isi kolom hilang dengan 0
    X = pd.DataFrame(columns=model_features)
    for c in model_features:
        X[c] = df[c] if c in df.columns else 0
    # Type safety
    for c in X.columns:
        if X[c].dtype == "O":
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0)

    preds = model.predict(X)
    probs = model.predict_proba(X)
    # Asumsikan order label sama seperti training: ['Dropout','Enrolled','Graduate'] (cek dengan label encoder saat training)
    # Kita cari indeks kelas 'Dropout' berdasarkan encoder
    classes = model.classes_ if hasattr(model, "classes_") else None
    # CalibratedClassifierCV expose classes_ di wrapper
    clf = model.named_steps["model"]
    class_names = clf.classes_
    try:
        dropout_idx = int(np.where(class_names == "Dropout")[0][0])
    except:
        # fallback: assume class order from training encoder
        dropout_idx = 0

    p_dropout = probs[:, dropout_idx]
    risk_level = np.where(p_dropout >= risk_thresh, "High",
                  np.where(p_dropout >= mid_thresh, "Medium", "Low"))
    out = df.copy()
    out["Predicted"] = class_names[preds]
    out["P(Dropout)"] = p_dropout
    out["Risk Level"] = risk_level
    return out, class_names

def main():
    st.title("Sistem Peringatan Dini Retensi Mahasiswa")
    st.markdown("Upload data mahasiswa untuk memprediksi risiko dropout dan melihat faktor-faktor yang berpengaruh.")

    model, label_enc, feat_imp = load_artifacts()

    col1, col2 = st.columns(2)
    with col1:
        risk_thresh = st.slider("Threshold High Risk (P(Dropout) ≥ ...)", 0.0, 1.0, RISK_THRESH_DEFAULT, 0.01)
    with col2:
        mid_thresh = st.slider("Threshold Medium Risk (P(Dropout) ≥ ...)", 0.0, 1.0, MID_THRESH_DEFAULT, 0.01)

    uploaded = st.file_uploader("Upload CSV (schema mirip dataset training)", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        df = clean_columns(df)

        # Jika ada Target, pisahkan agar tidak mempengaruhi prediksi
        if "Target" in df.columns:
            df_input = df.drop(columns=["Target"])
            st.info("Kolom Target terdeteksi dan diabaikan untuk inferensi.")
        else:
            df_input = df

        result, class_names = infer_and_flag(model, df_input, risk_thresh, mid_thresh)

        st.subheader("Hasil Prediksi")
        st.dataframe(result.style.background_gradient(subset=["P(Dropout)"], cmap="Reds"))

        # Ringkasan risiko
        st.subheader("Ringkasan Risiko")
        risk_counts = result["Risk Level"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).astype(int)
        c1, c2, c3 = st.columns(3)
        c1.metric("High risk", int(risk_counts.get("High", 0)))
        c2.metric("Medium risk", int(risk_counts.get("Medium", 0)))
        c3.metric("Low risk", int(risk_counts.get("Low", 0)))

        # Unduh
        st.download_button("Unduh Hasil CSV", data=result.to_csv(index=False), file_name="retention_predictions.csv", mime="text/csv")

        # Feature importance global
        st.subheader("Top fitur paling berpengaruh (global)")
        if feat_imp is not None:
            st.dataframe(feat_imp.head(15))
        else:
            st.write("Feature importance tidak tersedia.")

        # Penjelasan lokal sederhana: tampilkan top-k fitur input terbesar untuk beberapa contoh high risk
        st.subheader("Penjelasan lokal (indikasi faktor dominan)")
        k = st.slider("Tampilkan top K fitur per mahasiswa", 3, 10, 5)
        sample_high = result[result["Risk Level"] == "High"].head(5)
        if len(sample_high) > 0:
            st.caption("Catatan: ini pendekatan sederhana (fitur bernilai tinggi). Untuk eksplanasi lebih kuat, integrasikan SHAP di versi lanjutan.")
            for idx in sample_high.index:
                row = df_input.loc[idx]
                top_feats = row.abs().sort_values(ascending=False).head(k)
                st.write(f"Mahasiswa index {idx}: Predicted={result.loc[idx,'Predicted']}, P(Dropout)={result.loc[idx,'P(Dropout)']:.2f}")
                st.write(top_feats)
        else:
            st.write("Belum ada mahasiswa High risk untuk ditampilkan.")

    else:
        st.info("Silakan upload file CSV untuk memulai.")

if __name__ == "__main__":
    main()
