"""AUC inference, calibration and decision-curve utilities."""
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from .config import NONINF_MARGIN


# ----------------------------------------------------------------------------- AUC
def fast_auc(y, p):
    y = np.asarray(y)
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(np.asarray(p, float))
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bootstrap_auc(y, p, n_boot, rng):
    """AUC with percentile bootstrap 95% CI. Returns (auc, lo, hi, boot_values)."""
    y, p = np.asarray(y), np.asarray(p, float)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        a = fast_auc(y[i], p[i])
        if not np.isnan(a):
            vals.append(a)
    vals = np.array(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return fast_auc(y, p), lo, hi, vals


def _fast_delong(preds, m):
    """preds: (k, n) scores, first m columns are positives. Sun & Xu (2014)."""
    n = preds.shape[1] - m
    pos, neg = preds[:, :m], preds[:, m:]
    k = preds.shape[0]
    tx = np.vstack([rankdata(pos[r]) for r in range(k)])
    ty = np.vstack([rankdata(neg[r]) for r in range(k)])
    tz = np.vstack([rankdata(preds[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def delong_paired(y, p1, p2):
    """Two-sided paired DeLong test of AUC(p1) - AUC(p2) on the same patients."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="stable")
    m = int(y.sum())
    preds = np.vstack([np.asarray(p1, float)[order], np.asarray(p2, float)[order]])
    aucs, cov = _fast_delong(preds, m)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 1e-15:
        return diff, 1.0
    z = diff / np.sqrt(var)
    return diff, 2 * norm.sf(abs(z))


def paired_bootstrap_diff(y, p1, p2, n_boot, rng):
    """AUC(p1) - AUC(p2) with paired percentile bootstrap CI. Returns (diff, lo, hi)."""
    y, p1, p2 = np.asarray(y), np.asarray(p1, float), np.asarray(p2, float)
    n = len(y)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        a1, a2 = fast_auc(y[i], p1[i]), fast_auc(y[i], p2[i])
        if not (np.isnan(a1) or np.isnan(a2)):
            d.append(a1 - a2)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return fast_auc(y, p1) - fast_auc(y, p2), lo, hi


def compare_auc(y, p1, p2, rng, n_boot):
    """Difference (model 1 minus model 2), bootstrap CI, DeLong p, non-inferiority flag."""
    diff, lo, hi = paired_bootstrap_diff(y, p1, p2, n_boot, rng)
    _, p = delong_paired(y, p1, p2)
    return {"diff": diff, "ci_low": lo, "ci_high": hi, "p_delong": p,
            "non_inferior": bool(lo > -NONINF_MARGIN)}


def holm_adjust(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ------------------------------------------------------------------------ calibration
def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _expit(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def logistic_fit(x, y, ridge=1e-6, iters=100):
    """Intercept and slope of a logistic regression of y on x (Newton-Raphson)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    X = np.column_stack([np.ones(len(x)), x])
    b = np.zeros(2)
    for _ in range(iters):
        p = _expit(X @ b)
        g = X.T @ (y - p) - ridge * b
        H = (X.T * (p * (1 - p))) @ X + ridge * np.eye(2)
        step = np.linalg.solve(H, g)
        b += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return b


def calibration_summary(y, p):
    y, p = np.asarray(y), np.asarray(p, float)
    prev = y.mean()
    return {"n": len(y), "brier": float(np.mean((p - y) ** 2)),
            "brier_prevalence_only": float(prev * (1 - prev)),
            "calibration_slope": float(logistic_fit(_logit(p), y)[1]),
            "mean_predicted": float(p.mean()), "observed": float(prev)}


def calibration_bins(y, p, n_bins=5):
    """Mean predicted vs observed risk in quantile bins."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p, float)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    g = df.groupby("bin").agg(mean_predicted=("p", "mean"), observed=("y", "mean"))
    return g.reset_index(drop=True)


def recalibrate(y, p):
    """Logistic recalibration (intercept and slope) within the evaluation cohort.
    Optimistic by construction because it uses that cohort's outcomes."""
    b = logistic_fit(_logit(p), y)
    return _expit(b[0] + b[1] * _logit(p))


# --------------------------------------------------------------------- decision curve
def net_benefit(y, p, thresholds):
    y, p = np.asarray(y), np.asarray(p, float)
    n = len(y)
    out = []
    for t in thresholds:
        treat = p >= t
        tp = np.sum(treat & (y == 1))
        fp = np.sum(treat & (y == 0))
        out.append(tp / n - fp / n * t / (1 - t))
    return np.array(out)


def net_benefit_all(y, thresholds):
    prev = np.mean(y)
    t = np.asarray(thresholds)
    return prev - (1 - prev) * t / (1 - t)


def threshold_ranges(thresholds, mask):
    """Contiguous threshold ranges where mask is True, as a readable string."""
    thresholds = np.asarray(thresholds)
    ranges, start = [], None
    for i, ok in enumerate(mask):
        if ok and start is None:
            start = i
        if start is not None and (not ok or i == len(mask) - 1):
            end = i if ok else i - 1
            ranges.append((thresholds[start], thresholds[end]))
            start = None
    if not ranges:
        return "none"
    return "; ".join(f"{a*100:.0f}-{b*100:.0f}%" if a != b else f"{a*100:.0f}%"
                     for a, b in ranges)
