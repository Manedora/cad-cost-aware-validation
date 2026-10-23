"""Sensitivity analyses for missing data: complete-case analysis and multiple imputation."""
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

from .config import (DISCRETE, N_BOOT_PER_IMPUTATION, N_IMPUTATIONS, NONINF_MARGIN,
                     PREDICTORS, SEED)
from .stats_utils import fast_auc


def complete_case_mask(df, features):
    return df[features].notna().all(axis=1).to_numpy()


def mice_impute(df, with_outcome, n_imputations=N_IMPUTATIONS, seed=SEED):
    """Multiple imputation by chained equations within one external cohort.

    Bayesian ridge regression with posterior sampling; imputed values are clipped to the
    observed range and rounded for discrete variables. Optionally includes the outcome in
    the imputation model."""
    X = df[PREDICTORS].copy()
    if with_outcome:
        X["y"] = df["y"].to_numpy()
    # columns without variation (e.g. cholesterol set to one constant in the Swiss cohort)
    # cannot be modelled and are carried through unchanged
    lo_all, hi_all = X.min(), X.max()
    varying = [c for c in X.columns if hi_all[c] > lo_all[c]]
    Xv = X[varying]
    lo = lo_all[varying].to_numpy(float)
    hi = hi_all[varying].to_numpy(float)
    imputations = []
    for m in range(n_imputations):
        imp = IterativeImputer(estimator=BayesianRidge(), sample_posterior=True, max_iter=10,
                               min_value=lo, max_value=hi, keep_empty_features=True,
                               random_state=seed + m)
        out = X.copy()
        out[varying] = imp.fit_transform(Xv)
        out = out[PREDICTORS]
        for c in DISCRETE:
            out[c] = out[c].round()
        imputations.append(out)
    return imputations


def mi_evaluate(pipes, feature_sets, imputations, y, comparisons, rng,
                n_boot=N_BOOT_PER_IMPUTATION):
    """Pool AUCs and AUC differences over imputations.

    pipes / feature_sets : dict name -> fitted pipeline / feature list
    comparisons          : list of (name_a, name_b) -> difference a - b
    Point estimates are averaged over imputations; 95% CIs come from pooled bootstrap
    resamples (n_boot per imputation), using the same resample for all models (paired)."""
    y = np.asarray(y)
    n = len(y)
    names = list(pipes)
    point = {m: [] for m in names}
    boots = {m: [] for m in names}
    for imp in imputations:
        p = {m: pipes[m].predict_proba(imp[feature_sets[m]])[:, 1] for m in names}
        for m in names:
            point[m].append(fast_auc(y, p[m]))
        for _ in range(n_boot):
            i = rng.integers(0, n, n)
            vals = {m: fast_auc(y[i], p[m][i]) for m in names}
            if not any(np.isnan(v) for v in vals.values()):
                for m in names:
                    boots[m].append(vals[m])
    auc_rows, diff_rows = [], []
    for m in names:
        lo, hi = np.percentile(boots[m], [2.5, 97.5])
        auc_rows.append({"model": m, "auc": np.mean(point[m]), "ci_low": lo, "ci_high": hi})
    for a, b in comparisons:
        d = np.array(boots[a]) - np.array(boots[b])
        lo, hi = np.percentile(d, [2.5, 97.5])
        diff_rows.append({"comparison": f"{a} vs {b}",
                          "diff": np.mean(point[a]) - np.mean(point[b]),
                          "ci_low": lo, "ci_high": hi, "non_inferior": bool(lo > -NONINF_MARGIN)})
    return auc_rows, diff_rows
