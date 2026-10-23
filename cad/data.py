"""Loading and cleaning the four UCI Heart Disease cohorts."""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (COHORT_FILES, COLUMNS, PREDICTORS, TRAIN, ZERO_IS_MISSING,
                     ZERO_MISSING_COLUMNS)


def load_cohort(name: str, data_dir) -> pd.DataFrame:
    path = Path(data_dir) / COHORT_FILES[name]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python download_data.py` or place the four "
            "processed.*.data files from the UCI repository in the data directory.")
    df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
    df = df.apply(pd.to_numeric, errors="coerce")
    if name in ZERO_IS_MISSING:               # documented coding of missing values
        for col in ZERO_MISSING_COLUMNS:
            df.loc[df[col] == 0, col] = np.nan
    df["y"] = (df["num"] > 0).astype(int)     # label 1-4 = disease, 0 = no disease
    return df[PREDICTORS + ["y"]].reset_index(drop=True)


def load_all(data_dir):
    """Return (raw, filled).

    raw    : cohorts with missing values as recorded (used for missingness tables and
             complete-case analysis).
    filled : same, except that cholesterol, if never measured in a cohort (Swiss), is set
             to the Cleveland median in all other analyses (Methods).
    """
    raw = {c: load_cohort(c, data_dir) for c in COHORT_FILES}
    chol_median = raw[TRAIN]["chol"].median()
    filled = {}
    for c, df in raw.items():
        df = df.copy()
        if df["chol"].isna().all():
            df["chol"] = chol_median
        filled[c] = df
    return raw, filled
