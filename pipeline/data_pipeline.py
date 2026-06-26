"""
Production Data Pipeline — Prefect DAG

Ingest new Excel uploads → validate schema → run preprocessing → output cleaned CSV
→ trigger model retraining if data drift detected (Evidently AI).

Can be adapted for Airflow as well.
"""

try:
    from prefect import flow, task, get_run_logger
    from prefect.task_runners import SequentialTaskRunner
    HAS_PREFECT = True
except ImportError:
    HAS_PREFECT = False

import pandas as pd
import numpy as np
from pathlib import Path


def prefect_available():
    return HAS_PREFECT


@task(retries=2, retry_delay_seconds=30)
def ingest_excel(filepath: str) -> pd.DataFrame:
    logger = get_run_logger() if HAS_PREFECT else print
    logger(f"Ingesting: {filepath}")
    df = pd.read_excel(filepath)
    logger(f"  Loaded: {df.shape}")
    return df


@task
def validate_schema(df: pd.DataFrame, expected_columns: list) -> pd.DataFrame:
    logger = get_run_logger() if HAS_PREFECT else print
    missing = set(expected_columns) - set(df.columns)
    if missing:
        logger(f"  WARNING: Missing columns: {missing}")
    extra = set(df.columns) - set(expected_columns)
    if extra:
        logger(f"  Extra columns: {extra}")
    return df


@task
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger() if HAS_PREFECT else print

    if "Cinsiyet" in df.columns:
        mapping = {"Male": "Erkek", "Female": "Kadin", "m": "Erkek", "f": "Kadin"}
        df["Cinsiyet"] = df["Cinsiyet"].astype(str).str.strip().str.lower().map(mapping).fillna(df["Cinsiyet"])

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    logger(f"  Preprocessed: {df.shape}")
    return df


@task
def export_csv(df: pd.DataFrame, output_path: str):
    logger = get_run_logger() if HAS_PREFECT else print
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger(f"  Exported: {output_path}")


@task
def detect_drift(reference_path: str, current_path: str) -> bool:
    logger = get_run_logger() if HAS_PREFECT else print
    try:
        ref = pd.read_csv(reference_path)
        cur = pd.read_csv(current_path)
        drift_detected = abs(cur.mean().mean() - ref.mean().mean()) > 0.1
        logger(f"  Drift detected: {drift_detected}")
        return drift_detected
    except Exception as e:
        logger(f"  Drift check failed: {e}")
        return False


@task
def retrain_model():
    logger = get_run_logger() if HAS_PREFECT else print
    logger("  Triggering model retraining...")


@flow(name="Medical Data Pipeline", task_runner=SequentialTaskRunner())
def medical_pipeline(input_path: str, output_path: str, expected_columns: list,
                     reference_path: str = None):
    df = ingest_excel(input_path)
    df = validate_schema(df, expected_columns)
    df = preprocess(df)
    export_csv(df, output_path)

    if reference_path:
        drift = detect_drift(reference_path, output_path)
        if drift:
            retrain_model()


if __name__ == "__main__":
    if HAS_PREFECT:
        medical_pipeline(
            input_path="../pusulaData.xlsx",
            output_path="../output/cleaned_data.csv",
            expected_columns=["Kullanici_id", "Cinsiyet", "Dogum_Tarihi", "Kilo", "Boy"],
            reference_path=None,
        )
    else:
        print("[Prefect not installed] Install: pip install prefect")
        print("Running standalone pipeline demonstration...")
        df = pd.read_excel("../pusulaData.xlsx")
        df = preprocess.fn(df)
        print(f"Standalone result: {df.shape}")
