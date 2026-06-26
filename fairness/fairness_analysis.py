"""
Fairness Analysis — Model Performance Across Demographic Groups

Checks for disparate impact across Nationality, Gender, Age groups.
Computes per-group precision, recall, F1.

Uses Fairlearn library for measuring and mitigating disparate impact.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

try:
    from fairlearn.metrics import MetricFrame, selection_rate, false_positive_rate, true_positive_rate
    from fairlearn.postprocessing import ThresholdOptimizer
    HAS_FAIRLEARN = True
except ImportError:
    HAS_FAIRLEARN = False


def demographic_report(y_true, y_pred, sensitive_features, model_name="Model"):
    results = []
    for col in sensitive_features.columns:
        groups = sensitive_features[col].unique()
        for group in groups:
            mask = sensitive_features[col] == group
            if mask.sum() < 5:
                continue
            yt = y_true[mask]
            yp = y_pred[mask]
            results.append({
                "Feature": col,
                "Group": group,
                "N": mask.sum(),
                "Precision": precision_score(yt, yp, zero_division=0),
                "Recall": recall_score(yt, yp, zero_division=0),
                "F1": f1_score(yt, yp, zero_division=0),
                "Selection Rate": yp.mean(),
            })
    return pd.DataFrame(results)


def fairness_metrics(y_true, y_pred, sensitive_features):
    if not HAS_FAIRLEARN:
        print("[Fairlearn not installed] Install: pip install fairlearn")
        return {}

    metrics = {
        "F1": f1_score,
        "Precision": precision_score,
        "Recall": recall_score,
        "Selection Rate": selection_rate,
        "False Positive Rate": false_positive_rate,
        "True Positive Rate": true_positive_rate,
    }

    results = {}
    for col in sensitive_features.columns:
        sf = sensitive_features[col]
        metric_frame = MetricFrame(
            metrics=metrics,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sf,
        )
        results[col] = {
            "by_group": metric_frame.by_group,
            "difference": metric_frame.difference(),
            "ratio": metric_frame.ratio(),
        }
    return results


def mitigate_disparity(X, y, sensitive_features, model=None):
    if not HAS_FAIRLEARN:
        print("[Fairlearn not installed]")
        return None

    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    X_train, X_test, y_train, y_test, sf_train, sf_test = train_test_split(
        X, y, sensitive_features, test_size=0.3, random_state=42
    )

    post_model = ThresholdOptimizer(
        estimator=model,
        constraints="equalized_odds",
        prefit=False,
    )
    post_model.fit(X_train, y_train, sensitive_features=sf_train)

    y_pred_mitigated = post_model.predict(X_test, sensitive_features=sf_test)
    y_pred_original = model.fit(X_train, y_train).predict(X_test)

    return {
        "original_f1": f1_score(y_test, y_pred_original),
        "mitigated_f1": f1_score(y_test, y_pred_mitigated),
        "y_pred_original": y_pred_original,
        "y_pred_mitigated": y_pred_mitigated,
        "post_model": post_model,
    }


if __name__ == "__main__":
    print("Fairness Analysis — Medical Dataset")
    print("=" * 60)

    df = pd.read_excel("../pusulaData.xlsx")
    df["Cinsiyet"] = df["Cinsiyet"].fillna("Unknown")
    df["gender_binary"] = df["Cinsiyet"].map({"Male": "Male", "Female": "Female", "Erkek": "Male", "Kadin": "Female"}).fillna("Unknown")
    df["Yas"] = 2025 - pd.to_datetime(df["Dogum_Tarihi"], errors="coerce").dt.year
    df["Yas"] = df["Yas"].fillna(df["Yas"].median())
    df["age_group"] = pd.cut(df["Yas"], bins=[0, 30, 50, 70, 150], labels=["<30", "30-50", "50-70", "70+"])

    df["target"] = df["Kronik Hastaliklarim"].notna() & (df["Kronik Hastaliklarim"] != "")
    df["target"] = df["target"].astype(int)

    feature_cols = ["Kilo", "Boy"]
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
    df["Yas_scaled"] = (df["Yas"] - df["Yas"].mean()) / df["Yas"].std()
    for c in feature_cols:
        df[f"{c}_scaled"] = (df[c] - df[c].mean()) / df[c].std()

    X = df[["Yas_scaled", "Kilo_scaled", "Boy_scaled"]]
    y = df["target"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    sensitive = df.loc[X_test.index, ["gender_binary", "age_group"]]

    print("\nDemographic Performance Report:")
    report = demographic_report(y_test.values, y_pred, sensitive.reset_index(drop=True))
    print(report.to_string(index=False))
