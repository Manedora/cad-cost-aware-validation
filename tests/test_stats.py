"""Unit tests for the statistical helpers (run: python -m pytest tests or python tests/test_stats.py)."""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cad.stats_utils import (delong_paired, fast_auc, holm_adjust, logistic_fit,  # noqa: E402
                             net_benefit, net_benefit_all, paired_bootstrap_diff)
from cad.costs import verify_cost_table  # noqa: E402


def test_all():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    p1 = y * 0.8 + rng.normal(0, 1, 400)
    p2 = y * 0.3 + rng.normal(0, 1, 400)
    assert abs(fast_auc(y, p1) - roc_auc_score(y, p1)) < 1e-12
    d, p = delong_paired(y, p1, p1)
    assert d == 0 and p == 1.0
    d, p = delong_paired(y, p1, p2)
    assert abs(d - (roc_auc_score(y, p1) - roc_auc_score(y, p2))) < 1e-12 and p < 0.05
    # DeLong z agrees with a bootstrap standard error
    boots = [np.subtract(*[fast_auc(y[i], q[i]) for q in (p1, p2)])
             for i in (rng.integers(0, 400, 400) for _ in range(1000))]
    from scipy.stats import norm
    assert abs(2 * norm.sf(abs(d / np.std(boots))) - p) < 0.02
    # Holm
    assert np.allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    # logistic fit recovers slope
    x = rng.normal(size=5000); yy = (rng.random(5000) < 1 / (1 + np.exp(-(0.5 + 2 * x)))).astype(int)
    b = logistic_fit(x, yy)
    assert abs(b[0] - 0.5) < 0.15 and abs(b[1] - 2) < 0.2
    # net benefit
    yb = np.array([1, 1, 0, 0]); pb = np.array([.9, .6, .4, .1])
    assert abs(net_benefit(yb, pb, [0.5])[0] - 0.5) < 1e-12
    assert abs(net_benefit_all(yb, np.array([0.5]))[0] - 0.0) < 1e-12
    verify_cost_table()
    print("all tests passed")


if __name__ == "__main__":
    test_all()
