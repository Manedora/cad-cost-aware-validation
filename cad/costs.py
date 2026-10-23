"""Test costs (Turney) with group discounts, exhaustive subset search and Pareto front."""
from itertools import combinations, permutations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from .config import CV_FOLDS, PREDICTORS, SEED, SUBSET_CV_REPEATS
from .modeling import make_pipeline

# Costs in Canadian dollars (Ontario Health Insurance Program fee schedule, 1995), from the
# cost file distributed with the data (Turney 1995). Values are (full cost, discounted cost);
# the discounted cost applies to every further test of the same group once one has been done.
INDEPENDENT_COSTS = {"age": 1.00, "sex": 1.00, "cp": 1.00, "trestbps": 1.00, "restecg": 15.50}
GROUP_COSTS = {
    "blood_tests": {"chol": (7.27, 5.17), "fbs": (5.20, 3.10)},
    "exercise_ecg": {"exang": (87.30, 1.00), "oldpeak": (87.30, 1.00)},
    "thallium": {"thalach": (102.90, 1.00)},   # partner variable (thal) was excluded
}


def additive_cost(features):
    total = 0.0
    for f in features:
        if f in INDEPENDENT_COSTS:
            total += INDEPENDENT_COSTS[f]
        else:
            total += next(g[f][0] for g in GROUP_COSTS.values() if f in g)
    return round(total, 2)


def subset_cost(features):
    """Cost with group discounts: cheapest ordering of the tests within each group."""
    total = sum(INDEPENDENT_COSTS[f] for f in features if f in INDEPENDENT_COSTS)
    for group in GROUP_COSTS.values():
        members = [f for f in features if f in group]
        if not members:
            continue
        best = min(sum(group[f][0] if i == 0 else group[f][1] for i, f in enumerate(order))
                   for order in permutations(members))
        total += best
    return round(total, 2)


def verify_cost_table():
    """Cost values reported in the manuscript must be reproduced by the cost tables above."""
    checks = {
        "full": (PREDICTORS, 221.07, 309.47),
        "low-cost": (["age", "sex", "cp", "exang", "oldpeak"], 91.30, None),
        "screening": (["age", "sex", "cp", "trestbps"], 4.00, None),
        "best front subset": (["age", "sex", "cp", "restecg", "thalach", "exang", "oldpeak"],
                              209.70, None),
    }
    for name, (feats, disc, add) in checks.items():
        assert abs(subset_cost(feats) - disc) < 0.005, (name, subset_cost(feats), disc)
        if add is not None:
            assert abs(additive_cost(feats) - add) < 0.005, (name, additive_cost(feats), add)


def all_subsets(features=PREDICTORS):
    for k in range(1, len(features) + 1):
        for comb in combinations(features, k):
            yield list(comb)


def _eval_subset(features, df, splits):
    X, y = df[features], df["y"].to_numpy()
    aucs = []
    for tr, te in splits:
        pipe = make_pipeline("lr", features).fit(X.iloc[tr], y[tr])
        aucs.append(roc_auc_score(y[te], pipe.predict_proba(X.iloc[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs, ddof=1))


def exhaustive_search(train_df, n_jobs=-1):
    """Cross-validated AUC (repeated stratified 5-fold, 2 repeats) and cost of all 1,023 subsets."""
    cv = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=SUBSET_CV_REPEATS, random_state=SEED)
    splits = list(cv.split(train_df[PREDICTORS], train_df["y"]))
    subsets = list(all_subsets())
    res = Parallel(n_jobs=n_jobs)(delayed(_eval_subset)(s, train_df, splits) for s in subsets)
    out = pd.DataFrame({
        "features": [";".join(s) for s in subsets],
        "n_features": [len(s) for s in subsets],
        "cost": [subset_cost(s) for s in subsets],
        "additive_cost": [additive_cost(s) for s in subsets],
        "auc": [r[0] for r in res], "auc_sd": [r[1] for r in res]})
    out["pareto"] = pareto_flags(out)
    return out


def pareto_flags(df):
    """Pareto front: no strictly cheaper subset has an equal or higher AUC."""
    flags = []
    for _, r in df.iterrows():
        cheaper = df[df["cost"] < r["cost"] - 1e-9]
        flags.append(not (cheaper["auc"] >= r["auc"]).any())
    return flags


def select_low_cost(search, tolerance):
    """Cheapest subset whose AUC is within `tolerance` of the full model (ties: highest AUC)."""
    full_auc = search.loc[search["n_features"] == len(PREDICTORS), "auc"].iloc[0]
    cand = search[search["auc"] >= full_auc - tolerance - 1e-12]
    best = cand.sort_values(["cost", "auc"], ascending=[True, False]).iloc[0]
    full_cost = search.loc[search["n_features"] == len(PREDICTORS), "cost"].iloc[0]
    return {"tolerance": tolerance, "features": best["features"].split(";"),
            "cost": best["cost"], "auc": best["auc"],
            "saving_pct": 100 * (1 - best["cost"] / full_cost)}
