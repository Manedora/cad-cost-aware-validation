"""Preprocessing pipeline and the three classifiers."""
import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import BINARY, CATEGORICAL, CONTINUOUS, CV_FOLDS, SEED


def make_preprocessor(features):
    """Impute and scale/encode inside one pipeline; fitted on training data only."""
    # every non-categorical predictor (continuous and 0/1 variables) is median-imputed and
    # standardised; only chest pain type and resting ECG are one-hot encoded
    cont = [f for f in features if f in CONTINUOUS or f in BINARY]
    cat = [f for f in features if f in CATEGORICAL]
    parts = []
    if cont:
        parts.append(("cont", Pipeline([("imp", SimpleImputer(strategy="median")),
                                        ("scale", StandardScaler())]), cont))
    if cat:
        parts.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat))
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)


def make_classifier(name):
    if name == "lr":
        return LogisticRegression(max_iter=1000)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    if name == "gb":
        return GradientBoostingClassifier(random_state=SEED)   # scikit-learn defaults
    raise ValueError(name)


def make_pipeline(name, features):
    return Pipeline([("prep", make_preprocessor(features)), ("clf", make_classifier(name))])


def cv_predictions(name, features, df, n_splits=CV_FOLDS, seed=SEED):
    """Stratified k-fold CV in the training cohort.

    Returns out-of-fold probabilities and the mean of the fold-wise AUCs."""
    X, y = df[features], df["y"].to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(df))
    fold_auc = []
    for tr, te in skf.split(X, y):
        pipe = make_pipeline(name, features).fit(X.iloc[tr], y[tr])
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p
        fold_auc.append(roc_auc_score(y[te], p))
    return oof, float(np.mean(fold_auc))


def fit_and_predict_external(name, features, data, train="cleveland", external=None):
    """Fit on the training cohort; apply, without refitting, to every external cohort."""
    from .config import EXTERNAL
    external = external or EXTERNAL
    pipe = make_pipeline(name, features).fit(data[train][features], data[train]["y"])
    preds = {c: pipe.predict_proba(data[c][features])[:, 1] for c in external}
    return pipe, preds
