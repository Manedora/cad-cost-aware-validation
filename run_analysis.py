"""Run the complete analysis of the manuscript and write all tables and figures.

Usage:
    python download_data.py          # once, needs internet (or copy the 4 files into ./data)
    python run_analysis.py           # full analysis (as in the manuscript)
    python run_analysis.py --quick   # fewer bootstrap resamples; smoke test only
"""
import argparse
import importlib.metadata as md
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from cad import costs as C
from cad import plotting as P
from cad.config import *  # noqa: F401,F403
from cad.data import load_all
from cad.diamond_forrester import score_for
from cad.missing_data import complete_case_mask, mi_evaluate, mice_impute
from cad.modeling import cv_predictions, fit_and_predict_external, make_pipeline
from cad.stats_utils import (bootstrap_auc, calibration_summary, compare_auc, fast_auc,
                             holm_adjust, net_benefit, net_benefit_all, recalibrate,
                             threshold_ranges)


def log(msg):
    print(f"[{pd.Timestamp.now():%H:%M:%S}] {msg}", flush=True)


def fmt_ci(a, lo, hi, d=3):
    return f"{a:.{d}f} ({lo:.{d}f}-{hi:.{d}f})"


def main(data_dir, out_dir, quick=False):
    out = Path(out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    n_boot = 100 if quick else N_BOOT_AUC
    n_boot_pair = 200 if quick else N_BOOT_PAIRED
    n_imp = 3 if quick else N_IMPUTATIONS
    n_boot_imp = 20 if quick else N_BOOT_PER_IMPUTATION
    rng = np.random.default_rng(SEED)

    def save(df, name):
        df.to_csv(out / "tables" / f"{name}.csv", index=False)

    # ------------------------------------------------------------------ environment
    pkgs = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib", "shap"]
    env = {"python": platform.python_version()}
    for p in pkgs:
        try:
            env[p] = md.version(p)
        except md.PackageNotFoundError:
            env[p] = "not installed"
    (out / "environment.json").write_text(json.dumps(env, indent=2))
    log(f"environment: {env}")

    C.verify_cost_table()
    raw, data = load_all(data_dir)
    y = {c: data[c]["y"].to_numpy() for c in data}

    # --------------------------------------------- Table 1 and Supplementary Table S1
    rows = []
    for c in COHORT_FILES:
        d = raw[c]
        rows.append({"cohort": COHORT_LABELS[c], "N": len(d), "disease_pct": 100 * d["y"].mean(),
                     "mean_age": d["age"].mean(), "male_pct": 100 * d["sex"].mean()})
    save(pd.DataFrame(rows), "table1_cohorts")
    miss = pd.DataFrame({COHORT_LABELS[c]: 100 * raw[c][PREDICTORS].isna().mean() for c in raw})
    save(miss.reset_index().rename(columns={"index": "variable"}), "tableS1_missing_pct")

    # ------------------------------ Ten-predictor models: internal CV and external validation
    log("ten-predictor models")
    preds, oof, internal_auc, pipes = {}, {}, {}, {}
    for m in MODELS:
        oof[m], internal_auc[m] = cv_predictions(m, PREDICTORS, data[TRAIN])
        pipes[m], preds[m] = fit_and_predict_external(m, PREDICTORS, data)
    t2, t3 = [], []
    for m, label in MODELS.items():
        row = {"model": label, "internal_cv_auc": internal_auc[m]}
        for c in EXTERNAL:
            a, lo, hi, _ = bootstrap_auc(y[c], preds[m][c], n_boot, rng)
            row[COHORT_LABELS[c]] = fmt_ci(a, lo, hi)
        t2.append(row)
        for c, yy, pp in [(TRAIN, y[TRAIN], oof[m])] + [(c, y[c], preds[m][c]) for c in EXTERNAL]:
            t3.append({"model": label, "cohort": COHORT_LABELS[c], **calibration_summary(yy, pp)})
    save(pd.DataFrame(t2), "table2_auc_ten_predictors")
    save(pd.DataFrame(t3), "table3_calibration_all_models")   # logistic regression = Table 3

    # Supplementary Table S2: algorithm comparisons (6 tests, Holm)
    rows = []
    for c in EXTERNAL:
        for other in ("rf", "gb"):
            r = compare_auc(y[c], preds["lr"][c], preds[other][c], rng, n_boot_pair)
            rows.append({"cohort": COHORT_LABELS[c], "comparison": f"LR vs {other.upper()}", **r})
    s2 = pd.DataFrame(rows)
    s2["p_holm"] = holm_adjust(s2["p_delong"])
    save(s2, "tableS2_algorithm_comparisons")

    P.fig_roc({c: y[c] for c in EXTERNAL}, preds, out / "figures" / "fig2_roc.png")
    panels = {COHORT_LABELS[TRAIN] + " (out-of-fold)": (y[TRAIN], {m: oof[m] for m in MODELS})}
    for c in EXTERNAL:
        panels[COHORT_LABELS[c]] = (y[c], {m: preds[m][c] for m in MODELS})
    P.fig_calibration(panels, out / "figures" / "fig3_calibration.png")

    # ------------------------------------------------------------------------- SHAP
    try:
        import shap
        prep = pipes["lr"].named_steps["prep"]
        Xt = pd.DataFrame(prep.transform(data[TRAIN][PREDICTORS]),
                          columns=prep.get_feature_names_out())
        # use all training patients as background (shap subsamples to 100 by default)
        masker = shap.maskers.Independent(Xt, max_samples=len(Xt))
        sv = shap.LinearExplainer(pipes["lr"].named_steps["clf"], masker).shap_values(Xt)
        P.fig_shap(sv, Xt, out / "figures" / "fig4_shap_logistic_regression.png",
                   title="SHAP summary - logistic regression")
        P.fig_shap(sv, Xt, out / "figures" / "shap_bar_logistic_regression.png",
                   title="Mean |SHAP| - logistic regression", bar=True)
        prep_rf = pipes["rf"].named_steps["prep"]
        Xr = pd.DataFrame(prep_rf.transform(data[TRAIN][PREDICTORS]),
                          columns=prep_rf.get_feature_names_out())
        sv_rf = shap.TreeExplainer(pipes["rf"].named_steps["clf"]).shap_values(Xr)
        sv_rf = sv_rf[1] if isinstance(sv_rf, list) else (sv_rf[:, :, 1] if sv_rf.ndim == 3 else sv_rf)
        P.fig_shap(sv_rf, Xr, out / "figures" / "shap_summary_random_forest.png",
                   title="SHAP summary - random forest")
        P.fig_shap(sv_rf, Xr, out / "figures" / "shap_bar_random_forest.png",
                   title="Mean |SHAP| - random forest", bar=True)
        imp = pd.DataFrame({"feature": Xt.columns,
                            "mean_abs_shap_lr": np.abs(sv).mean(axis=0),
                            "mean_abs_shap_rf": np.abs(sv_rf).mean(axis=0)})
        save(imp.sort_values("mean_abs_shap_lr", ascending=False), "shap_importance")
    except ImportError:
        log("shap not installed: skipping Figure 4 (pip install shap)")

    # ------------------------------------------- Exhaustive subset search (Figure 5)
    log("exhaustive search over 1,023 subsets")
    search = C.exhaustive_search(data[TRAIN])
    save(search, "subset_search_all")
    save(search[search["pareto"]].sort_values("cost"), "pareto_front")
    sel = [C.select_low_cost(search, t) for t in LOW_COST_TOLERANCES]
    save(pd.DataFrame([{**s, "features": ";".join(s["features"])} for s in sel]),
         "low_cost_selection_by_tolerance")
    low_features = sel[0]["features"]
    full_row = search[search["n_features"] == len(PREDICTORS)].iloc[0]
    scr_row = search[search["features"] == ";".join(
        [f for f in PREDICTORS if f in SCREENING_FEATURES])].iloc[0]
    P.fig_pareto(search, [
        ("Full model", full_row["cost"], full_row["auc"], "*"),
        ("Low-cost model", sel[0]["cost"], sel[0]["auc"], "D"),
        ("Screening model", scr_row["cost"], scr_row["auc"], "s")], out / "figures" / "fig5_pareto.png")

    # ------------------------------------ External validation of cost-aware models
    log("cost-aware models: external validation")
    feats = {"full": PREDICTORS, "low-cost": low_features, "screening": SCREENING_FEATURES}
    ca_pipes, ca_preds, ca_oof = {}, {}, {}
    for k, f in feats.items():
        ca_pipes[k], ca_preds[k] = fit_and_predict_external("lr", f, data)
        ca_oof[k], _ = cv_predictions("lr", f, data[TRAIN])
    rows = []
    for k, f in feats.items():
        row = {"model": k, "features": ";".join(f), "cost": C.subset_cost(f)}
        for c in EXTERNAL:
            a, lo, hi, _ = bootstrap_auc(y[c], ca_preds[k][c], n_boot, rng)
            row[COHORT_LABELS[c]] = fmt_ci(a, lo, hi)
        rows.append(row)
    save(pd.DataFrame(rows), "table4_auc_cost_aware")

    pairs = [("low-cost", "full"), ("screening", "full"), ("screening", "low-cost")]
    rows = []
    for c in EXTERNAL:
        for a, b in pairs:
            r = compare_auc(y[c], ca_preds[a][c], ca_preds[b][c], rng, n_boot_pair)
            rows.append({"cohort": COHORT_LABELS[c], "comparison": f"{a} vs {b}", **r})
    t5 = pd.DataFrame(rows)
    t5["p_holm"] = holm_adjust(t5["p_delong"])          # 9 comparisons
    save(t5, "table5_paired_cost_aware")
    save(pd.DataFrame([{"model": k, "cohort": COHORT_LABELS[c], **calibration_summary(y[c], ca_preds[k][c])}
                       for k in feats for c in EXTERNAL]), "calibration_cost_aware")

    # ------------------------------------------------ Decision curve analysis (Fig 6)
    log("decision curve analysis")
    thr = np.round(np.arange(0.01, 1.0, 0.01), 2)
    curves, t6, ranges = {}, [], []
    for c in EXTERNAL:
        for kind in ("original", "recalibrated"):
            d = {"all": net_benefit_all(y[c], thr)}
            for k in feats:
                p = ca_preds[k][c] if kind == "original" else recalibrate(y[c], ca_preds[k][c])
                d[k] = net_benefit(y[c], p, thr)
                better = (d[k] > d["all"]) & (d[k] > 0)
                ranges.append({"cohort": COHORT_LABELS[c], "predictions": kind, "model": k,
                               "threshold_ranges_better_than_all_and_none":
                                   threshold_ranges(thr, better)})
            curves[(c, kind)] = d
        for t in DCA_THRESHOLDS_TABLE:
            i = int(np.argmin(np.abs(thr - t)))
            d = curves[(c, "original")]
            t6.append({"cohort": COHORT_LABELS[c], "threshold": t, "full": d["full"][i],
                       "low-cost": d["low-cost"][i], "screening": d["screening"][i],
                       "treat_all": d["all"][i]})
    save(pd.DataFrame(t6), "table6_net_benefit")
    save(pd.DataFrame(ranges), "dca_threshold_ranges")
    P.fig_dca(thr, curves, out / "figures" / "fig6_dca.png")

    # ----------------------------------------------------- Missing-data sensitivity
    log("missing-data sensitivity analyses")
    rows_auc, rows_diff, rows_cc = [], [], []
    comps = [("low-cost", "full"), ("screening", "full")]
    for c in EXTERNAL:
        # proportion with at least one missing predictor (as recorded)
        for k, f in feats.items():
            rows_cc.append({"cohort": COHORT_LABELS[c], "model": k,
                            "pct_with_missing_predictor": 100 * (~complete_case_mask(raw[c], f)).mean()})
        # primary analysis (median imputation with Cleveland values)
        for k in feats:
            a, lo, hi, _ = bootstrap_auc(y[c], ca_preds[k][c], n_boot, rng)
            rows_auc.append({"cohort": COHORT_LABELS[c], "method": "primary", "model": k,
                             "n": len(y[c]), "auc": a, "ci_low": lo, "ci_high": hi})
        for a_, b_ in comps:
            r = compare_auc(y[c], ca_preds[a_][c], ca_preds[b_][c], rng, n_boot_pair)
            rows_diff.append({"cohort": COHORT_LABELS[c], "method": "primary",
                              "comparison": f"{a_} vs {b_}", **r})
        # complete-case analysis (each model on its own complete cases; chol as recorded)
        for k, f in feats.items():
            mask = complete_case_mask(raw[c], f)
            yy = y[c][mask]
            if mask.sum() == 0 or len(np.unique(yy)) < 2:
                rows_auc.append({"cohort": COHORT_LABELS[c], "method": "complete case", "model": k,
                                 "n": int(mask.sum()), "auc": np.nan, "ci_low": np.nan, "ci_high": np.nan})
                continue
            pp = ca_pipes[k].predict_proba(raw[c].loc[mask, f])[:, 1]
            a, lo, hi, _ = bootstrap_auc(yy, pp, n_boot, rng)
            rows_auc.append({"cohort": COHORT_LABELS[c], "method": "complete case", "model": k,
                             "n": int(mask.sum()), "auc": a, "ci_low": lo, "ci_high": hi})
        # multiple imputation, with and without the outcome in the imputation model
        for label, with_y in [("MI, predictors only", False), ("MI, with outcome", True)]:
            imps = mice_impute(data[c], with_y, n_imputations=n_imp, seed=SEED)
            aucs, diffs = mi_evaluate({k: ca_pipes[k] for k in feats}, feats, imps, y[c], comps,
                                      rng, n_boot=n_boot_imp)
            for r in aucs:
                rows_auc.append({"cohort": COHORT_LABELS[c], "method": label, "model": r["model"],
                                 "n": len(y[c]), "auc": r["auc"], "ci_low": r["ci_low"],
                                 "ci_high": r["ci_high"]})
            for r in diffs:
                rows_diff.append({"cohort": COHORT_LABELS[c], "method": label, **r})
    save(pd.DataFrame(rows_cc), "missing_share_by_model")
    save(pd.DataFrame(rows_auc), "tableS3_auc_by_missing_data_method")
    save(pd.DataFrame(rows_diff), "table7_difference_by_missing_data_method")

    # --------------------------------------------- Diamond-Forrester comparison
    log("Diamond-Forrester comparison")
    rows8, rows9, rowsc = [], [], []
    df_pred, y_df, mp = {}, {}, {}
    for c in COHORT_FILES:
        s = score_for(data[c])
        mask = ~np.isnan(s)
        df_pred[c], y_df[c], mp[c] = s[mask], y[c][mask], mask
    rows_all = []
    for c in COHORT_FILES:
        model_p = {k: (ca_oof[k] if c == TRAIN else ca_preds[k][c])[mp[c]] for k in feats}
        yy, s = y_df[c], df_pred[c]
        a, lo, hi, _ = bootstrap_auc(yy, s, n_boot, rng)
        row8 = {"cohort": COHORT_LABELS[c] + (" (out-of-fold)" if c == TRAIN else ""),
                "n": len(yy), "n_disease": int(yy.sum()), "n_no_disease": int((1 - yy).sum()),
                "pct_defined": 100 * mp[c].mean(), "prevalence_in_subset": yy.mean(),
                "diamond_forrester": fmt_ci(a, lo, hi)}
        for k in feats:
            a2, lo2, hi2, _ = bootstrap_auc(yy, model_p[k], n_boot, rng)
            row8[k] = fmt_ci(a2, lo2, hi2)
            r = compare_auc(yy, model_p[k], s, rng, n_boot_pair)
            rows_all.append({"cohort": COHORT_LABELS[c], "model": k, **r})
        rows8.append(row8)
        rowsc.append({"cohort": COHORT_LABELS[c], **calibration_summary(yy, s)})
    t9 = pd.DataFrame(rows_all)
    t9["p_holm"] = holm_adjust(t9["p_delong"])           # 12 comparisons
    save(pd.DataFrame(rows8), "table8_auc_diamond_forrester")
    save(t9, "table9_difference_vs_diamond_forrester")
    save(pd.DataFrame(rowsc), "calibration_diamond_forrester")

    log(f"finished; results in {out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--quick", action="store_true",
                    help="fewer bootstrap resamples/imputations (smoke test, NOT for reporting)")
    a = ap.parse_args()
    main(a.data_dir, a.out_dir, a.quick)
