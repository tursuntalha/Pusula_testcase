"""
MedClean — Reusable Medical Data Preprocessing Library

Clean API:
  MedClean(df).detect_anomalies().expand_multilabel('My_Diseases').impute().encode()

Publishes to PyPI: pip install medclean
"""

import re
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler


class MedClean:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._label_encoders = {}
        self._scaler = None

    def detect_anomalies(self, id_col=None, gender_col=None, nationality_col=None):
        if id_col and id_col in self.df.columns:
            orig = self.df[id_col].copy()
            self.df[f"{id_col}_original"] = orig
            self.df[id_col] = pd.factorize(orig.astype(str))[0] + 1

        if gender_col and gender_col in self.df.columns:
            mapping = {
                "male": "Erkek", "female": "Kadin",
                "m": "Erkek", "f": "Kadin",
                "erkek": "Erkek", "kadin": "Kadin",
            }
            self.df[gender_col] = (
                self.df[gender_col]
                .astype(str).str.strip().str.lower()
                .map(mapping).fillna(self.df[gender_col])
            )

        if nationality_col and nationality_col in self.df.columns:
            if self.df[nationality_col].nunique() <= 1:
                self.df = self.df.drop(columns=[nationality_col])

        return self

    def extract_date_features(self, date_cols=None):
        if date_cols is None:
            date_cols = [c for c in self.df.columns if self.df[c].dtype.kind == 'M']

        for col in date_cols:
            if col not in self.df.columns:
                continue
            dt = pd.to_datetime(self.df[col], errors='coerce')
            base = col.replace(" ", "_").lower()
            self.df[f"{base}_year"] = dt.dt.year
            self.df[f"{base}_month"] = dt.dt.month
            self.df[f"{base}_day"] = dt.dt.day
            self.df[f"{base}_weekday"] = dt.dt.weekday

        return self

    def expand_multilabel(self, column, sep=","):
        if column not in self.df.columns:
            return self

        values = self.df[column].fillna("")
        all_labels = set()
        for v in values:
            for label in str(v).split(sep):
                label = label.strip()
                if label and label.lower() not in ("nan", "", " "):
                    all_labels.add(label)

        all_labels = sorted(all_labels)
        for label in all_labels:
            self.df[f"{column}_{label.strip()}"] = values.apply(
                lambda x, l=label: 1 if l in [t.strip() for t in str(x).split(sep)] else 0
            )

        return self

    def impute(self, numeric_strategy="knn", categorical_strategy="most_frequent",
               n_neighbors=5):
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(include=["object", "category"]).columns.tolist()

        if categorical_strategy and cat_cols:
            imp_cat = SimpleImputer(strategy=categorical_strategy)
            self.df[cat_cols] = imp_cat.fit_transform(self.df[cat_cols])

        if numeric_strategy == "knn" and len(num_cols) > 1:
            imp_num = KNNImputer(n_neighbors=n_neighbors)
            self.df[num_cols] = imp_num.fit_transform(self.df[num_cols])
        elif numeric_strategy and num_cols:
            imp_num = SimpleImputer(strategy=numeric_strategy)
            self.df[num_cols] = imp_num.fit_transform(self.df[num_cols])

        return self

    def encode(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=["object", "category"]).columns.tolist()

        for col in columns:
            if col not in self.df.columns:
                continue
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self._label_encoders[col] = le

        return self

    def scale(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if columns:
            self._scaler = StandardScaler()
            self.df[columns] = self._scaler.fit_transform(self.df[columns])

        return self

    def get(self):
        return self.df


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_excel("../pusulaData.xlsx")

    cleaned = (
        MedClean(df)
        .detect_anomalies(id_col="Kullanici_id", gender_col="Cinsiyet", nationality_col="Uyruk")
        .extract_date_features()
        .expand_multilabel("Kronik Hastaliklarim")
        .impute()
        .encode()
        .scale()
        .get()
    )
    print(f"MedClean output: {cleaned.shape}")
    print(f"Columns: {cleaned.columns.tolist()}")
