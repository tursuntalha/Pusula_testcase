# Pusula Data Science Case Study — Medical Data Preprocessing & Anomaly Detection

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
</p>

A data preprocessing and anomaly detection case study on a medical/health dataset containing patient records with disease history, demographics, and timestamps. Developed as a technical assessment for **Pusula**.

**Project Vision:** A real-world data quality challenge. Healthcare data is notoriously messy — inconsistent ID formats, multi-label disease columns, mixed data types, and culturally-diverse categorical values (nationalities, gender encoding conventions). The anomaly detection and multi-label disease imputation techniques demonstrated here are directly applicable to production healthcare data cleaning pipelines, clinical trial preprocessing, and any data engineering role where you inherit dirty data and need to deliver a model-ready dataset without dropping rows.

---

## Dataset Overview

The dataset (`pusulaData.xlsx`) contains patient-level records with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `User_id` | String/Int | Unique patient identifier |
| `Nationality` | Categorical | Patient nationality |
| `My_Diseases` | Multi-label string | Comma-separated list of diseases |
| `datetime64` | Datetime | Record timestamp |
| `Gender` | Categorical | Patient gender |

---

## What Was Done

| Step | Description |
|------|-------------|
| **Anomaly Detection** | Identified malformed data types, invalid `User_id` formats, and inconsistencies in `Nationality` and `Gender` columns |
| **Missing Value Analysis** | Per-column missing value counts and visual inspection using Matplotlib/Seaborn |
| **Disease Column Expansion** | Multi-label `My_Diseases` string expanded into individual binary indicator columns for each disease |
| **Disease Imputation** | Missing disease entries inferred from inter-disease correlations (e.g., co-occurring conditions) |
| **Categorical Imputation** | `SimpleImputer` with most-frequent strategy for `Nationality`, `Gender` |
| **Numeric Imputation** | `KNNImputer` for numerical columns to preserve local data structure |
| **Encoding** | `LabelEncoder` applied to all categorical columns for model readiness |
| **Scaling** | `StandardScaler` applied as the final preprocessing step |

---

## Pipeline

```
pusulaData.xlsx
      │
      ▼
Data Analysis
  ─ Data type inspection
  ─ Missing value per column
  ─ Distribution plots
      │
      ▼
Anomaly Correction
  ─ Fix User_id formats
  ─ Standardize Gender / Nationality labels
      │
      ▼
Feature Engineering
  ─ Expand My_Diseases → binary columns
  ─ Extract datetime components (year, month, day, weekday)
      │
      ▼
Imputation
  ─ Disease columns: correlation-based fill
  ─ Categorical: SimpleImputer (most frequent)
  ─ Numeric: KNNImputer
      │
      ▼
Encoding & Scaling
  ─ LabelEncoder → StandardScaler
      │
      ▼
Model-Ready Dataset
```

---

## Project Structure

```
Pusula_testcase/
├── caseCode.ipynb        # Main notebook: full preprocessing pipeline
├── pusulaData.xlsx       # Raw dataset
├── documentation.txt     # Detailed project documentation (Turkish)
└── README.md
```

---

## Setup & Run

```bash
pip install pandas numpy scikit-learn openpyxl matplotlib seaborn jupyter

jupyter notebook caseCode.ipynb
```

Run all cells in order. The notebook is self-contained and produces a cleaned, encoded, scaled dataframe ready for downstream modeling.

> Detailed methodology is documented in `documentation.txt` (Turkish).

---

## Beyond the Case Study

The preprocessing pipeline here is solid — the next step is turning it into reusable tooling and applying real ML:

- [ ] **MedClean Python Library** — Package the preprocessing steps (anomaly detection, multi-label expansion, correlation-based imputation) as a reusable Python library with a clean API: `MedClean(df).detect_anomalies().expand_multilabel('My_Diseases').impute().encode()`. Publish to PyPI.
- [ ] **ML Model Comparison** — Apply classification models on the cleaned dataset to predict disease risk or patient outcomes. Compare: Logistic Regression (baseline), Random Forest, XGBoost, and a simple MLP. Use stratified K-fold CV and report F1, ROC-AUC, and precision-recall curves for each.
- [ ] **Fairness Analysis** — Check if any trained model performs differently across demographic groups (Nationality, Gender). Compute per-group precision, recall, and F1. Use the Fairlearn library to measure and mitigate disparate impact. Healthcare models must not discriminate.
- [ ] **Uncertainty Quantification** — Replace point predictions with prediction sets using conformal prediction (`MAPIE` library). Output: "This patient has an 80% prediction interval of [disease_risk: 0.3–0.6]." More informative and safer for clinical decision support.
- [ ] **Production Data Pipeline** — Redesign the notebook as a scheduled Prefect or Airflow DAG: ingest new Excel uploads → validate schema → run preprocessing → output cleaned CSV → trigger model retraining if data drift is detected (Evidently AI).
- [ ] **SHAP Disease Explanation** — After training a model, use SHAP to explain which features drive disease risk predictions per patient. This is the difference between a model and a clinical tool a doctor could actually use.
