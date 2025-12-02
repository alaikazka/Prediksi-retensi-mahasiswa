# train.py
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.inspection import permutation_importance
import joblib

RANDOM_STATE = 42

# Kolom target dan normalisasi nama kolom yang sering bermasalah
TARGET_CANDIDATES = ["Target", "Target ", " target"]
GRADE1_CANDIDATES = ["Curricular units 1st sem (grade)", "Curricular units 1st sem (grade) "]
GRADE2_CANDIDATES = ["Curricular units 2nd sem (grade)", " Curricular units 2nd sem (grade) "]

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Hilangkan whitespace dan backslash
    df.columns = [c.strip().replace("\\", "").replace("  ", " ").strip() for c in df.columns]
    # Perbaikan spesifik
    rename_map = {}
    # Target
    for cand in TARGET_CANDIDATES:
        if cand in df.columns:
            rename_map[cand] = "Target"
            break
    # Grade 1
    for cand in GRADE1_CANDIDATES:
        if cand in df.columns:
            rename_map[cand] = "Curricular units 1st sem (grade)"
            break
    # Grade 2
    for cand in GRADE2_CANDIDATES:
        if cand in df.columns:
            rename_map[cand] = "Curricular units 2nd sem (grade)"
            break
    # Course with trailing backslash
    if "Course \\" in df.columns:
        rename_map["Course \\"] = "Course"
    # Application mode sometimes split
    if "Application" in df.columns and "mode" in df.columns and "Application mode" not in df.columns:
        # If accidentally split, try to rebuild (optional)
        pass

    df = df.rename(columns=rename_map)
    return df

def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = clean_columns(df)
    # Pastikan tidak ada missing di kolom numerik (dataset kamu memang 0, tapi berjaga-jaga)
    for c in df.columns:
        if c != "Target":
            if df[c].dtype == "O":
                # Jika ada string kosong pada kolom numerik, coba convert
                try:
                    df[c] = pd.to_numeric(df[c], errors="ignore")
                except:
                    pass
    return df

def get_features_target(df: pd.DataFrame):
    if "Target" not in df.columns:
        raise ValueError("Kolom Target tidak ditemukan setelah pembersihan.")
    X = df.drop(columns=["Target"])
    y = df["Target"].astype(str).str.strip()
    return X, y

def build_pipeline(n_estimators=100, max_depth=None, min_samples_split=5):
    # Semua fitur numerik dibiarkan apa adanya (no scaling untuk RF)
    numeric_cols = FunctionTransformer(lambda X: X.columns.tolist()).fit_transform
    # Model utama
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    # Kalibrasi probabilitas
    calibrated = CalibratedClassifierCV(rf, method="isotonic", cv=3)
    # Pipeline sederhana (tanpa transform khusus karena semua numerik)
    pipe = Pipeline(steps=[
        ("model", calibrated)
    ])
    return pipe

def main():
    # Path dataset
    data_path = "dataset.csv"
    df = load_dataset(data_path)
    X, y = get_features_target(df)

    # Encode label
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
    )

    # Build & fit
    pipe = build_pipeline(n_estimators=100, max_depth=None, min_samples_split=5)
    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    print(f"Akurasi: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"Macro Recall: {rec:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("\nConfusion Matrix (label order same as LabelEncoder):")
    print(confusion_matrix(y_test, y_pred))

    # Global feature importance via permutation importance (lebih reliabel daripada RF feature_importances_)
    rf_estimator = pipe.named_steps["model"].base_estimator  # RandomForestClassifier
    rf_estimator.fit(X_train, y_train)  # fit ulang base untuk importance
    perm = permutation_importance(rf_estimator, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std
    }).sort_values("importance_mean", ascending=False)
    print("\nTop 10 fitur paling berpengaruh:")
    print(importance_df.head(10))

    # Simpan artefak
    Path("artifacts").mkdir(exist_ok=True)
    joblib.dump(pipe, "artifacts/retention_model.joblib")
    joblib.dump(le, "artifacts/label_encoder.joblib")
    importance_df.to_csv("artifacts/feature_importance.csv", index=False)
    print("\nModel dan artefak disimpan di folder 'artifacts'.")

if __name__ == "__main__":
    main()
