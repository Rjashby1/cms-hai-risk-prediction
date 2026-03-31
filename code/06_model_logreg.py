import numpy as np
from scipy.stats import loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline


def build_logreg_pipeline(
    preprocess,
    penalty="elasticnet",
    class_weight=None,
    random_seed=42,
):
    """
    Build a Logistic Regression modeling pipeline.

    Parameters
    ----------
    preprocess : sklearn transformer or pipeline
        Preprocessing object to apply before model fitting.
    penalty : {"elasticnet", "l1", "l2"}, default="elasticnet"
        Regularization type for Logistic Regression.
    class_weight : dict, str, or None, default=None
        Class weighting strategy.
    random_seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    sklearn Pipeline
        Pipeline containing preprocessing and Logistic Regression.
    """
    if penalty == "elasticnet":
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            class_weight=class_weight,
            max_iter=10000,
            random_state=random_seed,
        )

    elif penalty == "l1":
        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            class_weight=class_weight,
            max_iter=10000,
            random_state=random_seed,
        )

    elif penalty == "l2":
        model = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            class_weight=class_weight,
            max_iter=10000,
            random_state=random_seed,
        )

    else:
        raise ValueError("Invalid penalty. Choose from 'elasticnet', 'l1', or 'l2'.")

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("logreg", model),
    ])

    return pipe


def define_logreg_grid(penalty="elasticnet"):
    """
    Define the hyperparameter search space for Logistic Regression.

    Parameters
    ----------
    penalty : {"elasticnet", "l1", "l2"}, default="elasticnet"
        Regularization type to tune.

    Returns
    -------
    dict
        Parameter search space for RandomizedSearchCV.
    """
    if penalty == "elasticnet":
        params = {
            "logreg__C": loguniform(1e-4, 1e4),
            "logreg__l1_ratio": np.linspace(0.0, 1.0, 10),
        }

    elif penalty in ["l1", "l2"]:
        params = {
            "logreg__C": loguniform(1e-4, 1e4),
        }

    else:
        raise ValueError("Invalid penalty. Choose from 'elasticnet', 'l1', or 'l2'.")

    return params


def tune_logreg_model(
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
    Tune a Logistic Regression pipeline using RandomizedSearchCV.

    Parameters
    ----------
    pipe : sklearn Pipeline
        Pipeline containing preprocessing and Logistic Regression.
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
    logreg_cv = RandomizedSearchCV(
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

    logreg_cv.fit(Xtrain, ytrain)
    return logreg_cv


def run_logreg_model(
    preprocess,
    cv,
    Xtrain,
    ytrain,
    penalty="elasticnet",
    class_weight=None,
    random_seed=42,
    n_iter=50,
    scoring="roc_auc",
    n_jobs=-1,
):
    """
    Run the full Logistic Regression workflow: build, tune, and fit.

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
    penalty : {"elasticnet", "l1", "l2"}, default="elasticnet"
        Logistic Regression penalty type.
    class_weight : dict, str, or None, default=None
        Class weighting strategy.
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
    pipe = build_logreg_pipeline(
        preprocess=preprocess,
        penalty=penalty,
        class_weight=class_weight,
        random_seed=random_seed,
    )

    params = define_logreg_grid(penalty=penalty)

    logreg_cv = tune_logreg_model(
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

    return logreg_cv, pipe, params