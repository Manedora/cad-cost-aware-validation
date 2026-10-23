"""Constants shared by all analysis modules."""

SEED = 42  # fixed random seed (Methods: "The random seed was fixed (42)")

COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
           "exang", "oldpeak", "slope", "ca", "thal", "num"]

# Ten retained predictors (slope, ca and thal were excluded: up to 99% missing externally)
PREDICTORS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
              "thalach", "exang", "oldpeak"]
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]   # median-imputed + standardised
BINARY = ["sex", "fbs", "exang"]                                  # 0/1; handled like continuous variables
CATEGORICAL = ["cp", "restecg"]                                   # most-frequent imputed + one-hot
DISCRETE = BINARY + CATEGORICAL                                   # rounded after multiple imputation

COHORT_FILES = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}
COHORT_LABELS = {"cleveland": "Cleveland", "hungarian": "Hungarian",
                 "switzerland": "Swiss", "va": "VA Long Beach"}
TRAIN = "cleveland"
EXTERNAL = ["hungarian", "switzerland", "va"]
# Cohorts in which zero cholesterol / blood pressure means "not recorded"
ZERO_IS_MISSING = {"switzerland", "va"}
ZERO_MISSING_COLUMNS = ["chol", "trestbps"]

MODELS = {"lr": "Logistic regression", "rf": "Random forest", "gb": "Gradient boosting"}

# Analysis settings taken from the Methods
N_BOOT_AUC = 1000          # bootstrap resamples for AUC CIs
N_BOOT_PAIRED = 2000       # paired bootstrap for AUC differences
N_IMPUTATIONS = 20         # multiple imputation (MICE)
N_BOOT_PER_IMPUTATION = 100
NONINF_MARGIN = 0.05       # exploratory non-inferiority margin (AUC)
CV_FOLDS = 5
SUBSET_CV_REPEATS = 2
LOW_COST_TOLERANCES = [0.01, 0.005, 0.02]   # primary first
SCREENING_FEATURES = ["age", "sex", "cp", "trestbps"]  # the four CA$1 variables
DCA_THRESHOLDS_TABLE = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
