"""
ML Model Comparison — Classification on Cleaned Medical Dataset

Models:
  - Logistic Regression (baseline)
  - Random Forest
  - XGBoost
  - Simple MLP

Evaluation: Stratified K-fold CV, F1, ROC-AUC, Precision-Recall curves
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    precision_recall_curve, roc_curve,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

import warnings
warnings.filterwarnings("ignore")


def compare_models(X, y, cv_folds=5):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced"),
        "MLP (2-layer)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                                          use_label_encoder=False, eval_metric="logloss")

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    results = []
    for name, model in models.items():
        fold_metrics = {"f1": [], "roc_auc": [], "precision": [], "recall": []}

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            y_proba = model.predict_proba(X_test_s)[:, 1]

            fold_metrics["f1"].append(f1_score(y_test, y_pred))
            fold_metrics["roc_auc"].append(roc_auc_score(y_test, y_proba))
            fold_metrics["precision"].append(precision_score(y_test, y_pred))
            fold_metrics["recall"].append(recall_score(y_test, y_pred))

        results.append({
            "Model": name,
            "F1 (mean±std)": f"{np.mean(fold_metrics['f1']):.4f}±{np.std(fold_metrics['f1']):.4f}",
            "ROC-AUC (mean±std)": f"{np.mean(fold_metrics['roc_auc']):.4f}±{np.std(fold_metrics['roc_auc']):.4f}",
            "Precision (mean±std)": f"{np.mean(fold_metrics['precision']):.4f}±{np.std(fold_metrics['precision']):.4f}",
            "Recall (mean±std)": f"{np.mean(fold_metrics['recall']):.4f}±{np.std(fold_metrics['recall']):.4f}",
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("ML Model Comparison — Medical Dataset")
    print("=" * 60)

    df = pd.read_excel("../pusulaData.xlsx")

    num_cols = ["Kilo", "Boy"]
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    df["Cinsiyet_int"] = df["Cinsiyet"].map({"Male": 1, "Female": 0, "Erkek": 1, "Kadin": 0}).fillna(0)
    df["Yas"] = 2025 - pd.to_datetime(df["Dogum_Tarihi"], errors="coerce").dt.year
    df["Yas"] = df["Yas"].fillna(df["Yas"].median())

    disease_mask = df["Kronik Hastaliklarim"].notna() & (df["Kronik Hastaliklarim"] != "")
    df["has_disease"] = disease_mask.astype(int)

    feature_cols = ["Cinsiyet_int", "Yas", "Kilo", "Boy"]
    X = df[feature_cols].fillna(0)
    y = df["has_disease"]

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target distribution: {y.value_dict() if hasattr(y, 'value_dict') else y.value_counts().to_dict()}")
    print()

    results = compare_models(X, y)
    print(results.to_string(index=False))
