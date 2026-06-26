# MedClean — Medical Data Preprocessing Library

```python
from medclean import MedClean

cleaned = MedClean(df).detect_anomalies().expand_multilabel('My_Diseases').impute().encode().get()
```

## Installation

```bash
pip install medclean
```

## API

| Method | Description |
|--------|-------------|
| `detect_anomalies()` | Fix ID formats, standardize gender/nationality |
| `extract_date_features()` | Extract year/month/day/weekday from datetime cols |
| `expand_multilabel(col)` | Expand comma-separated multi-label column into binary indicators |
| `impute()` | KNNImputer for numeric, SimpleImputer for categorical |
| `encode()` | LabelEncode all categorical columns |
| `scale()` | StandardScaler all numeric columns |
| `get()` | Return cleaned DataFrame |
