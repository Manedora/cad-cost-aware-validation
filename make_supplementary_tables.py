"""Supplementary Tables S4-S6 (predictor definitions, distributions by cohort, model coefficients)
and one workbook with S1-S6. Run after run_analysis.py (needs results/tables/*.csv)."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cad.config import (BINARY, COHORT_FILES, COHORT_LABELS, CONTINUOUS, PREDICTORS,
                        SCREENING_FEATURES, TRAIN)
from cad.costs import subset_cost
from cad.data import load_all
from cad.modeling import fit_and_predict_external

DEFINITIONS = [
    ("age", "Age", "years", "continuous"),
    ("sex", "Sex", "1 = male, 0 = female", "binary"),
    ("cp", "Chest pain type", "1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic", "categorical"),
    ("trestbps", "Resting systolic blood pressure on admission", "mm Hg (0 coded as missing in Swiss and VA cohorts)", "continuous"),
    ("chol", "Serum cholesterol", "mg/dl (0 coded as missing in Swiss and VA cohorts)", "continuous"),
    ("fbs", "Fasting blood sugar > 120 mg/dl", "1 = true, 0 = false", "binary"),
    ("restecg", "Resting electrocardiogram", "0 = normal, 1 = ST-T wave abnormality, 2 = probable or definite left ventricular hypertrophy", "categorical"),
    ("thalach", "Maximum heart rate achieved", "beats per minute", "continuous"),
    ("exang", "Exercise-induced angina", "1 = yes, 0 = no", "binary"),
    ("oldpeak", "ST depression induced by exercise relative to rest", "mm", "continuous"),
]


def table_s4():
    rows = []
    for v, desc, coding, kind in DEFINITIONS:
        rows.append({"variable": v, "description": desc, "unit / coding": coding, "type": kind,
                     "single-test cost (CA$)": subset_cost([v]),
                     "preprocessing": ("median imputation + standardisation" if kind != "categorical"
                                       else "most-frequent imputation + one-hot encoding")})
    return pd.DataFrame(rows)


def table_s5(raw):
    out = []
    for v in PREDICTORS:
        if v in CONTINUOUS:
            row = {"variable": v, "level": "mean (SD)"}
            for c, d in raw.items():
                row[COHORT_LABELS[c]] = f"{d[v].mean():.1f} ({d[v].std():.1f})" if d[v].notna().any() else "not measured"
            out.append(row)
        else:
            levels = sorted(raw[TRAIN][v].dropna().unique()) if v not in BINARY else [1]
            for lv in levels:
                row = {"variable": v, "level": f"= {int(lv)}, n (%)"}
                for c, d in raw.items():
                    obs = d[v].dropna()
                    row[COHORT_LABELS[c]] = f"{int((obs == lv).sum())} ({100 * (obs == lv).mean():.1f})" if len(obs) else "not measured"
                out.append(row)
        miss = {COHORT_LABELS[c]: f"{100 * d[v].isna().mean():.1f}" for c, d in raw.items()}
        out.append({"variable": v, "level": "missing, %", **miss})
    n = {COHORT_LABELS[c]: str(len(d)) for c, d in raw.items()}
    dis = {COHORT_LABELS[c]: f"{int(d['y'].sum())} ({100 * d['y'].mean():.1f})" for c, d in raw.items()}
    return pd.DataFrame([{"variable": "N", "level": "patients", **n},
                         {"variable": "outcome", "level": "angiographic CAD, n (%)", **dis}] + out)


def table_s6(data, models):
    rows = []
    for name, feats in models.items():
        pipe, _ = fit_and_predict_external("lr", feats, data)
        prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
        coefs = dict(zip(prep.get_feature_names_out(), clf.coef_[0]))
        rows.append({"model": name, "term": "intercept", "coefficient (log-odds)": clf.intercept_[0],
                     "centre (training median, used for imputation)": "", "mean used for standardisation": "",
                     "SD used for standardisation": "", "coefficient per raw unit": ""})
        cont_feats = [f for f in feats if f in CONTINUOUS or f in BINARY]
        cont = prep.named_transformers_["cont"] if cont_feats else None
        for f in feats:
            if f in cont_feats:
                i = cont_feats.index(f)
                med = cont.named_steps["imp"].statistics_[i]
                mu, sd = cont.named_steps["scale"].mean_[i], cont.named_steps["scale"].scale_[i]
                rows.append({"model": name, "term": f, "coefficient (log-odds)": coefs[f],
                             "centre (training median, used for imputation)": med,
                             "mean used for standardisation": mu, "SD used for standardisation": sd,
                             "coefficient per raw unit": coefs[f] / sd})
            else:
                for term, val in coefs.items():
                    if term.startswith(f + "_"):
                        rows.append({"model": name, "term": term, "coefficient (log-odds)": val,
                                     "centre (training median, used for imputation)": "", "mean used for standardisation": "",
                                     "SD used for standardisation": "", "coefficient per raw unit": ""})
    df = pd.DataFrame(rows)
    num = df.select_dtypes("number").columns
    df[num] = df[num].round(4)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()
    out = Path(a.out_dir) / "tables"
    raw, data = load_all(a.data_dir)
    low = pd.read_csv(out / "low_cost_selection_by_tolerance.csv").iloc[0]["features"].split(";")
    models = {"Full (10 predictors)": PREDICTORS,
              "Low-cost": [f for f in PREDICTORS if f in low],
              "Screening": [f for f in PREDICTORS if f in SCREENING_FEATURES]}
    s4, s5, s6 = table_s4(), table_s5(raw), table_s6(data, models)
    s4.to_csv(out / "tableS4_predictor_definitions.csv", index=False)
    s5.to_csv(out / "tableS5_predictors_by_cohort.csv", index=False)
    s6.to_csv(out / "tableS6_model_coefficients.csv", index=False)
    with pd.ExcelWriter(Path(a.out_dir) / "Supplementary_tables.xlsx") as xw:
        for name, df in [("S1 missing data", pd.read_csv(out / "tableS1_missing_pct.csv").round(1)),
                         ("S2 algorithm comparisons", pd.read_csv(out / "tableS2_algorithm_comparisons.csv").round(3)),
                         ("S3 AUC by missing-data method", pd.read_csv(out / "tableS3_auc_by_missing_data_method.csv").round(3)),
                         ("S4 predictor definitions", s4), ("S5 predictors by cohort", s5),
                         ("S6 model coefficients", s6)]:
            df.to_excel(xw, sheet_name=name[:31], index=False)
    print("written to", out)
