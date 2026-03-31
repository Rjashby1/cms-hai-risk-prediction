from pathlib import Path

import joblib
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline


def build_rf_pipeline(
    preprocess,
    class_weight="balanced",
    random_seed=42,
):
    """
    Build a Random Forest modeling pipeline.

    Parameters
    ----------
    preprocess : sklearn transformer or pipeline
        Preprocessing object to apply before model fitting.
    class_weight : str, dict, or None, default="balanced"
        Class weighting strategy.
    random_seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    sklearn Pipeline
        Pipeline containing preprocessing and Random Forest.
    """
    model = RandomForestClassifier(
        class_weight=class_weight,
        random_state=random_seed,
        n_jobs=-1,
    )

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("rf", model),
    ])

    return pipe


def define_rf_grid():
    """
    Define the hyperparameter search space for Random Forest.

    Returns
    -------
    dict
        Parameter search space for RandomizedSearchCV.
    """
    params = {
        "rf__n_estimators": randint(100, 500),
        "rf__max_depth": [None, 5, 10, 20, 30],
        "rf__min_samples_split": randint(2, 11),
        "rf__min_samples_leaf": randint(1, 6),
        "rf__max_features": ["sqrt", "log2"],
    }

    return params


def tune_rf_model(
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
    Tune a Random Forest pipeline using RandomizedSearchCV.

    Parameters
    ----------
    pipe : sklearn Pipeline
        Pipeline containing preprocessing and Random Forest.
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
    rf_cv = RandomizedSearchCV(
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

    rf_cv.fit(Xtrain, ytrain)
    return rf_cv


def run_rf_model(
    preprocess,
    cv,
    Xtrain,
    ytrain,
    class_weight="balanced",
    random_seed=42,
    n_iter=50,
    scoring="roc_auc",
    n_jobs=-1,
):
    """
    Run the full Random Forest workflow: build, tune, and fit.

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
    class_weight : str, dict, or None, default="balanced"
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
    pipe = build_rf_pipeline(
        preprocess=preprocess,
        class_weight=class_weight,
        random_seed=random_seed,
    )

    params = define_rf_grid()

    rf_cv = tune_rf_model(
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

    return rf_cv, pipe, params


def save_model(model, filename="randforest_model.joblib"):
    """
    Save a trained model object to the project's models directory.

    Parameters
    ----------
    model : object
        Fitted model or pipeline to save.
    filename : str, default="randforest_model.joblib"
        Output filename.

    Returns
    -------
    pathlib.Path
        Full save path of the stored model file.
    """
    out_dir = Path(__file__).parent / "../models"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_path = out_dir / filename
    joblib.dump(model, save_path)
    print(f"[SUCCESS] Model saved to {save_path}")

    return save_path