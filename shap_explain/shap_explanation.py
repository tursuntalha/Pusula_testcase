"""
SHAP Disease Explanation — Feature Importance for Disease Risk

Uses SHAP to explain which features drive disease risk predictions per patient.
This is the difference between a model and a clinical tool doctors can use.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def explain_disease_risk(X, y, feature_names=None):
    if not HAS_SHAP:
        print("[SHAP not installed] Install: pip install shap")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    print("SHAP Disease Explanation")
    print(f"  Model accuracy: {model.score(X_test, y_test):.3f}")
    print(f"  SHAP values shape: {np.array(shap_values).shape}")
    print()

    if isinstance(shap_values, list):
        class_idx = 1
        sv = shap_values[class_idx]
    else:
        sv = shap_values

    feature_importance = pd.DataFrame({
        "feature": feature_names if feature_names else X.columns,
        "importance": np.abs(sv).mean(axis=0),
    }).sort_values("importance", ascending=False)

    print("Feature Importance (mean |SHAP value|):")
    print(feature_importance.to_string(index=False))

    try:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        shap.summary_plot(sv, X_test, feature_names=feature_names, show=False)
        plt.title("SHAP Summary Plot")

        plt.subplot(1, 2, 2)
        shap.bar_plot(sv, X_test, feature_names=feature_names, show=False)
        plt.title("SHAP Feature Importance")

        plt.tight_layout()
        plt.savefig("shap_disease_explanation.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("\nPlot saved: shap_disease_explanation.png")
    except Exception as e:
        print(f"\nPlotting error: {e}")

    return {
        "shap_values": sv,
        "feature_importance": feature_importance,
        "explainer": explainer,
        "model": model,
    }


if __name__ == "__main__":
    print("SHAP Disease Explanation — Medical Dataset")
    print("=" * 60)

    df = pd.read_excel("../pusulaData.xlsx")

    df["Cinsiyet_int"] = df["Cinsiyet"].map({"Male": 1, "Female": 0, "Erkek": 1, "Kadin": 0}).fillna(0)
    df["Yas"] = 2025 - pd.to_datetime(df["Dogum_Tarihi"], errors="coerce").dt.year
    df[["Kilo", "Boy"]] = df[["Kilo", "Boy"]].fillna(df[["Kilo", "Boy"]].median())
    df["BMI"] = df["Kilo"] / ((df["Boy"] / 100) ** 2)

    feature_cols = ["Cinsiyet_int", "Yas", "Kilo", "Boy", "BMI"]
    X = df[feature_cols].fillna(0)
    y = (
        df["Kronik Hastaliklarim"].notna()
        & (df["Kronik Hastaliklarim"] != "")
    ).astype(int)

    explain_disease_risk(X, y, feature_names=feature_cols)
