from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import numpy as np
import pandas as pd
import sklearn
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DEFAULT_DATA = Path(__file__).resolve().parent / "train.csv"
DEFAULT_MODEL = Path(__file__).resolve().parent / "pipeline.bin"

TARGET = "price_range"
FEATURES = [
    "battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g",
    "int_memory", "m_dep", "mobile_wt", "n_cores", "pc", "px_height",
    "px_width", "ram", "sc_h", "sc_w", "talk_time", "three_g",
    "touch_screen", "wifi"
]


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Removes rows containing anomalous zero values and statistical outliers.

    This function filters the dataset by performing two cleaning steps:
    1. Drops rows with physically impossible zero values in screen dimensions
       (`px_height` and `sc_w`).
    2. Drops rows containing statistical outliers in continuous numerical columns
       using the Interquartile Range (1.5 * IQR) rule.

    Args:
        df (pd.DataFrame): Raw input DataFrame containing mobile phone features
            (e.g., `train.csv`).

    Returns:
        pd.DataFrame: A filtered copy of the input DataFrame with anomalous rows
            removed and a reset index.
    """
    df_clean = df.copy()

    # Drop rows with physical zero anomalies in screen dimensions
    zero_anomaly_cols = ["px_height", "sc_w"]
    for col in zero_anomaly_cols:
        if col in df_clean.columns:
            df_clean = df_clean[df_clean[col] > 0]

    # Drop rows with statistical outliers using the 1.5 * IQR rule
    exclude_cols = {"price_range", "id"}
    num_cols = [
        col for col in df_clean.select_dtypes(include=[np.number]).columns
        if df_clean[col].nunique() > 2 and col not in exclude_cols
    ]

    # Build a combined boolean mask across all continuous columns
    valid_mask = pd.Series(True, index=df_clean.index)
    for col in num_cols:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        valid_mask &= (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)

    df_clean = df_clean[valid_mask].reset_index(drop=True)
    return df_clean


def build_pipeline(
    c_param: float = 10.0,  # Update this if your GridSearchCV winner used C=1.0 or C=50.0
    kernel: str = "linear",
    gamma: str = "scale",
):
    """Builds the winning StandardScaler + SVC classification pipeline."""
    return make_pipeline(
        StandardScaler(),
        SVC(
            C=c_param,
            kernel=kernel,
            gamma=gamma,
            probability=True,
            random_state=42,
        ),
    )


def train(
    data_path: Path,
    model_path: Path,
    c_param: float = 10.0,  # Update this to match your GridSearchCV winner
    kernel: str = "linear",
    gamma: str = "scale",
) -> None:
    """Loads raw data, cleans anomalies, evaluates via 5-Fold CV & hold-out test, and exports artifacts."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    # 1. Load and clean dataset
    raw_df = pd.read_csv(data_path)
    clean_df = preprocessing(raw_df)

    X = clean_df[FEATURES]
    y = clean_df[TARGET].astype(int)

    # 2. Reproducible 90/10 split (Train+Val 90% vs Hold-out Test 10%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42, stratify=y
    )

    # 3. Evaluate via 5-Fold Stratified CV and hold-out test for reproducibility logging
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    eval_pipeline = build_pipeline(c_param=c_param, kernel=kernel, gamma=gamma)

    cv_scores = cross_val_score(
        eval_pipeline, X_train_val, y_train_val, cv=cv_strategy, scoring="roc_auc_ovr", n_jobs=-1
    )

    eval_pipeline.fit(X_train_val, y_train_val)
    test_probs = eval_pipeline.predict_proba(X_test)
    test_preds = eval_pipeline.predict(X_test)
    test_auc = roc_auc_score(y_test, test_probs, multi_class="ovr")
    test_acc = accuracy_score(y_test, test_preds)

    # 4. Fit final production pipeline on full cleaned dataset
    final_pipeline = build_pipeline(c_param=c_param, kernel=kernel, gamma=gamma)
    final_pipeline.fit(X, y)

    # Ensure output directory exists and serialize model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as model_file:
        pickle.dump(final_pipeline, model_file, protocol=pickle.HIGHEST_PROTOCOL)

    # 5. Compute SHA-256 hashes & export feature defaults (medians for inference fallback)
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()

    defaults = {col: float(X[col].median()) for col in FEATURES}
    defaults_path = model_path.with_name("feature_defaults.json")
    defaults_path.write_text(json.dumps(defaults, indent=2) + "\n", encoding="utf-8")
    defaults_hash = hashlib.sha256(defaults_path.read_bytes()).hexdigest()

    # 6. Export reproducible metadata artifact
    metadata = {
        "dataset": data_path.name,
        "dataset_sha256": data_hash,
        "raw_rows": len(raw_df),
        "clean_rows": len(clean_df),
        "rows_removed": len(raw_df) - len(clean_df),
        "features": FEATURES,
        "target": TARGET,
        "model": f"StandardScaler + SVC(C={c_param}, kernel={kernel}, gamma={gamma}, probability=True)",
        "validation_strategy": "90/10 Stratified Split + 5-Fold Stratified CV",
        "random_state": 42,
        "scikit_learn": sklearn.__version__,
        "preprocessing": "Drop px_height==0, sc_w==0, and 1.5*IQR outliers",
        "metrics": {
            "cv_auc_ovr_mean": round(float(cv_scores.mean()), 6),
            "cv_auc_ovr_std": round(float(cv_scores.std()), 6),
            "test_auc_ovr": round(float(test_auc), 6),
            "test_accuracy": round(float(test_acc), 6),
        },
        "model_sha256": model_hash,
        "feature_defaults_sha256": defaults_hash,
    }

    metadata_path = model_path.with_name("model_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the Mobile Price Classification SVC model for ML Zoomcamp 2026 midterm project"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to train.csv")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Output path for pipeline.bin")
    parser.add_argument("--C", type=float, default=10.0, help="Regularization parameter")
    parser.add_argument("--kernel", type=str, default="linear", help="Kernel type (linear, rbf, poly)")
    parser.add_argument("--gamma", type=str, default="scale", help="Kernel coefficient (scale, auto)")
    args = parser.parse_args()

    train(
        args.data,
        args.model,
        c_param=args.C,
        kernel=args.kernel,
        gamma=args.gamma,
    )
