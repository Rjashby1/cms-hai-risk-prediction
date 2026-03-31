import numpy as np
from scipy.stats import loguniform, randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def build_xgb_pipeline(
    preprocess,
    scale_pos_weight=None,
    random_seed=42,
):
    """
    Build an XGBoost modeling pipeline.

    Parameters
    ----------
    preprocess : sklearn transformer or pipeline
        Preprocessing object to apply before model fitting.
    scale_pos_weight : float or None, default=None
        Positive-class weighting factor used to address class imbalance.
    random_seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    sklearn Pipeline
        Pipeline containing preprocessing and XGBoost.
    """
    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=random_seed,
        n_jobs=1,
        verbosity=0,
    )

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("xgb", model),
    ])

    return pipe


def define_xgb_grid():
    """
    Define the hyperparameter search space for XGBoost.

    Returns
    -------
    dict
        Parameter search space for RandomizedSearchCV.
    """
    params = {
        "xgb__n_estimators": randint(100, 500),
        "xgb__learning_rate": loguniform(0.01, 0.3),
        "xgb__max_depth": randint(3, 10),
        "xgb__min_child_weight": randint(1, 6),
        "xgb__subsample": np.linspace(0.6, 1.0, 5),
        "xgb__colsample_bytree": np.linspace(0.6, 1.0, 5),
    }

    return params


def tune_xgb_model(
    pipe,
    params,
    cv,
    Xtrain,
    ytrain,
    n_iter=50,
    scoring="roc_auc",
    random_seed=42,
    n_jobs=-1,
):
    """
    Tune an XGBoost pipeline using RandomizedSearchCV.

    Parameters
    ----------
    pipe : sklearn Pipeline
        Pipeline containing preprocessing and XGBoost.
    params : dict
        Hyperparameter search space.
    cv : cross-validation splitter
        Cross-validation object such as StratifiedKFold.
    Xtrain : array-like or DataFrame
        Training features.
    ytrain : array-like or Series
        Training labels.
    n_iter : int, default=50
        Number of random search iterations.
    scoring : str, default="roc_auc"
        Model selection metric.
    random_seed : int, default=42
        Random seed for reproducibility.
    n_jobs : int, default=-1
        Number of parallel jobs.

    Returns
    -------
    RandomizedSearchCV
        Fitted randomized search object.
    """
    xgb_cv = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=params,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_seed,
        n_jobs=n_jobs,
        verbose=2,
        return_train_score=True,
        refit=True,
    )

    xgb_cv.fit(Xtrain, ytrain)
    return xgb_cv


def run_xgb_model(
    preprocess,
    cv,
    Xtrain,
    ytrain,
    scale_pos_weight=None,
    random_seed=42,
    n_iter=50,
    scoring="roc_auc",
    n_jobs=-1,
):
    """
    Run the full XGBoost workflow: build, tune, and fit.

    Parameters
    ----------
    preprocess : sklearn transformer or pipeline
        Preprocessing object to apply before model fitting.
    cv : cross-validation splitter
        Cross-validation object such as StratifiedKFold.
    Xtrain : array-like or DataFrame
        Training features.
    ytrain : array-like or Series
        Training labels.
    scale_pos_weight : float or None, default=None
        Positive-class weighting factor used to address class imbalance.
    random_seed : int, default=42
        Random seed for reproducibility.
    n_iter : int, default=50
        Number of random search iterations.
    scoring : str, default="roc_auc"
        Model selection metric.
    n_jobs : int, default=-1
        Number of parallel jobs.

    Returns
    -------
    tuple
        (fitted RandomizedSearchCV object, pipeline, parameter grid)
    """
    pipe = build_xgb_pipeline(
        preprocess=preprocess,
        scale_pos_weight=scale_pos_weight,
        random_seed=random_seed,
    )

    params = define_xgb_grid()

    xgb_cv = tune_xgb_model(
        pipe=pipe,
        params=params,
        cv=cv,
        Xtrain=Xtrain,
        ytrain=ytrain,
        n_iter=n_iter,
        scoring=scoring,
        random_seed=random_seed,
        n_jobs=n_jobs,
    )

    return xgb_cv, pipe, params