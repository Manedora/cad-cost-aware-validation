"""Figures 2-6 of the manuscript (Figure 1 is a study-flow schematic drawn separately)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from .config import COHORT_LABELS, EXTERNAL, MODELS  # noqa: E402
from .stats_utils import calibration_bins, fast_auc  # noqa: E402


def fig_roc(y_ext, preds, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, c in zip(axes, EXTERNAL):
        for m, label in MODELS.items():
            fpr, tpr, _ = roc_curve(y_ext[c], preds[m][c])
            ax.plot(fpr, tpr, label=f"{label} (AUC {fast_auc(y_ext[c], preds[m][c]):.2f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set(title=COHORT_LABELS[c], xlabel="1 - specificity", ylabel="Sensitivity")
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_calibration(panels, path):
    """panels: dict label -> (y, {model: p})"""
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for ax, (label, (y, ps)) in zip(np.atleast_1d(axes), panels.items()):
        for m in ("lr", "rf"):
            b = calibration_bins(y, ps[m], 5)
            ax.plot(b["mean_predicted"], b["observed"], "o-", label=MODELS[m])
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set(title=label, xlabel="Mean predicted probability",
               ylabel="Observed proportion", xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_pareto(search, marks, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(search["cost"], search["auc"], s=8, c="lightgrey", label="All subsets")
    front = search[search["pareto"]].sort_values("cost")
    ax.step(front["cost"], front["auc"], where="post", c="red", label="Pareto front")
    for (label, cost, auc, marker) in marks:
        ax.scatter([cost], [auc], marker=marker, s=90, edgecolor="k", zorder=5, label=label)
    ax.set_xscale("log")
    ax.set(xlabel="Test cost (CA$, log scale)", ylabel="Cross-validated AUC")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_dca(thresholds, curves, path):
    """curves: dict (cohort, 'original'|'recalibrated') -> {'all', model: nb arrays}"""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for j, c in enumerate(EXTERNAL):
        for i, kind in enumerate(("original", "recalibrated")):
            ax = axes[i, j]
            d = curves[(c, kind)]
            ax.plot(thresholds, d["all"], c="grey", label="Treat all")
            ax.axhline(0, ls="--", c="k", lw=0.8, label="Treat none")
            for m in ("full", "low-cost", "screening"):
                ax.plot(thresholds, d[m], label=m)
            lo = min(0.0, min(np.min(d[m]) for m in ("full", "low-cost", "screening")))
            ax.set_ylim(max(lo, -0.1), max(np.max(d["all"]), np.max(d["full"])) * 1.1)
            ax.set_title(f"{COHORT_LABELS[c]} ({kind})")
            ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_shap(shap_values, X_df, path, title=None, bar=False):
    import shap
    plt.figure()
    if bar:
        shap.summary_plot(shap_values, X_df, plot_type="bar", show=False)
    else:
        shap.summary_plot(shap_values, X_df, show=False)
    if title:
        plt.title(title)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
