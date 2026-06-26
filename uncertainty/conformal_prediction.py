"""
Uncertainty Quantification — Conformal Prediction with MAPIE

Outputs prediction intervals for disease risk:
  "This patient has an 80% prediction interval of [disease_risk: 0.3–0.6]"
Provides safer clinical decision support than point predictions.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

try:
    from mapie.classification import MapieClassifier
    from mapie.regression import MapieRegressor
    from mapie.metrics import regression_coverage_score
    HAS_MAPIE = True
except ImportError:
    HAS_MAPIE = False


def conformal_classification(X, y, alpha=0.2, method="score"):
    if not HAS_MAPIE:
        print("[MAPIE not installed] Install: pip install mapie")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    mapie = MapieClassifier(estimator=base_model, method=method, cv=5)
    mapie.fit(X_train, y_train)

    y_pred, y_pis = mapie.predict(X_test, alpha=alpha)

    coverage = np.mean([y_test.values[i] in y_pis[i, :, 0] for i in range(len(y_test))])
    avg_set_size = np.mean([len(y_pis[i, :, 0]) for i in range(len(y_test))])

    print(f"Conformal Classification (alpha={alpha})")
    print(f"  Coverage: {coverage:.3f}")
    print(f"  Avg prediction set size: {avg_set_size:.2f}")

    return {"y_pred": y_pred, "y_pis": y_pis, "mapie": mapie, "coverage": coverage}


def conformal_regression(X, y, alpha=0.2):
    if not HAS_MAPIE:
        print("[MAPIE not installed]")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    from sklearn.ensemble import RandomForestRegressor
    base_model = RandomForestRegressor(n_estimators=100, random_state=42)
    mapie = MapieRegressor(estimator=base_model, cv=5)
    mapie.fit(X_train, y_train)

    y_pred, y_pis = mapie.predict(X_test, alpha=alpha)

    coverage = regression_coverage_score(y_test, y_pis[:, 0, 0], y_pis[:, 1, 0])
    avg_width = np.mean(y_pis[:, 1, 0] - y_pis[:, 0, 0])

    print(f"Conformal Regression (alpha={alpha})")
    print(f"  Coverage: {coverage:.3f}")
    print(f"  Avg interval width: {avg_width:.3f}")
    print(f"  Sample output: risk={y_pred[0]:.3f} ± {(y_pis[0, 1, 0] - y_pis[0, 0, 0])/2:.3f}")

    return {"y_pred": y_pred, "y_pis": y_pis, "mapie": mapie, "coverage": coverage}


if __name__ == "__main__":
    print("Uncertainty Quantification — Conformal Prediction")
    print("=" * 60)

    df = pd.read_excel("../pusulaData.xlsx")

    df["Cinsiyet_int"] = df["Cinsiyet"].map({"Male": 1, "Female": 0, "Erkek": 1, "Kadin": 0}).fillna(0)
    df["Yas"] = 2025 - pd.to_datetime(df["Dogum_Tarihi"], errors="coerce").dt.year
    df[["Kilo", "Boy"]] = df[["Kilo", "Boy"]].fillna(df[["Kilo", "Boy"]].median())

    X = df[["Cinsiyet_int", "Yas", "Kilo", "Boy"]].fillna(0)
    y = (
        df["Kronik Hastaliklarim"].notna()
        & (df["Kronik Hastaliklarim"] != "")
    ).astype(int)

    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target prevalence: {y.mean():.2%}")

    conformal_classification(X, y, alpha=0.2)
