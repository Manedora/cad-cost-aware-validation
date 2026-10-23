# Code for: cost-aware, externally validated machine-learning models for angiographic CAD

This repository reproduces every table and figure (except the schematic Figure 1) of the manuscript
from the four public UCI Heart Disease cohorts.

## Quick start
```bash
pip install -r requirements.txt     # pin the exact versions you used, see environment.json
python download_data.py             # downloads the 4 processed.*.data files into ./data
python run_analysis.py              # full analysis (long: the 1,023-subset search dominates)
```
Outputs: `results/tables/*.csv`, `results/figures/*.png`, `results/environment.json`
(exact Python and package versions of the run, to be quoted in the Methods).

`python make_supplementary_tables.py` (after `run_analysis.py`) writes Supplementary Tables S4-S6 (predictor definitions, distributions by cohort, model coefficients) and `results/Supplementary_tables.xlsx` with S1-S6.
`python run_analysis.py --quick` uses fewer bootstrap resamples and imputations and is a smoke test only.
Figure 4 (SHAP) needs `pip install shap`; it is skipped with a message if shap is missing. Its code path could not be executed in the reference run, but the equivalent stand-alone script ran without errors on the authors' machine and reproduced the calibration results.
`python tests/test_stats.py` checks the statistical helpers (AUC, DeLong, Holm, calibration slope,
net benefit) and that the cost tables reproduce the costs reported in the paper.
`tests/make_synthetic_data.py` writes meaningless synthetic files for testing without the real data.

## Reproducibility notes
* Point estimates (AUCs, Brier scores, slopes, costs, net benefit) are deterministic given the seed (42).
* Bootstrap confidence intervals and multiple-imputation results depend on the random-number stream, which
  differs slightly across NumPy/scikit-learn versions; re-running may change the third decimal of some
  intervals and, for the VA multiple-imputation analysis, absolute AUCs by up to about 0.01.
  Conclusions are unchanged.
* The reference run took about 17 minutes on a single CPU core.

## License
MIT (see `LICENSE`). The UCI Heart Disease data are public (CC BY 4.0) and are not redistributed here.

## Map from manuscript to code
| Manuscript | Code |
|---|---|
| Outcome, 10 predictors, zeros = missing in Swiss/VA, Swiss cholesterol | `cad/data.py`, `cad/config.py` |
| Single preprocessing pipeline, LR / RF (300 trees) / GB, stratified 5-fold CV, seed 42 | `cad/modeling.py` |
| AUC bootstrap CI, DeLong, paired bootstrap, Holm, non-inferiority (0.05), calibration, DCA | `cad/stats_utils.py` |
| Test costs with group discounts, exhaustive search of 1,023 subsets, Pareto front, low-cost/screening models | `cad/costs.py` |
| Complete-case analysis and MICE (20 imputations, Bayesian ridge) | `cad/missing_data.py` |
| Diamond-Forrester pretest probability | `cad/diamond_forrester.py` |
| Figures 2-6 | `cad/plotting.py` |
| Tables 1-9, S1-S3 and orchestration | `run_analysis.py` |

Table files: `table1_cohorts`, `table2_auc_ten_predictors`, `table3_calibration_all_models` (logistic
regression rows = Table 3), `table4_auc_cost_aware`, `table5_paired_cost_aware`, `table6_net_benefit`,
`table7_difference_by_missing_data_method`, `table8_auc_diamond_forrester`,
`table9_difference_vs_diamond_forrester`, `tableS1_missing_pct`, `tableS2_algorithm_comparisons`,
`tableS3_auc_by_missing_data_method`.

## Implementation choices not fully specified in the manuscript
* All predictors except chest pain type and resting ECG (including the 0/1 variables sex, fasting blood sugar and exercise angina) are median-imputed and standardised; this choice reproduces the manuscript's numbers.
* Cost values are stored in `cad/costs.py` (Turney's cost file); `verify_cost_table()` checks that they reproduce
  the costs reported in the paper (CA$221.07 full, 309.47 additive, 91.30 low-cost, 4.00 screening).
* Diamond-Forrester probabilities are in `cad/diamond_forrester.py` and must be checked against the original table.
* Random seeds of bootstraps and imputations derive from seed 42; small numerical differences from the
  manuscript are possible across scikit-learn/NumPy versions.
* Figure 1 (study flow) is a schematic and is not produced by the code.
   ## Authors
   - Zahra Gandomi (z.gandomi@iau.ir)
   - Dariush Rastegari
