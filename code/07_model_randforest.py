from pathlib import Path

import joblib
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline


def build_rf_pipeline(preprocess, class_weight="balanced", random_seed=42):
    """
    Build a Random Forest pipeline with preprocessing.

    Parameters:
    preprocess: preprocessing pipeline to apply to the data
    class_weight (str or dict, optional): class weighting strategy
    random_seed (int, optional): random seed for reproducibility

    Returns:
    Pipeline: preprocessing + Random Forest pipeline
    """

    pipe = Pipeline(
        [
            ("preprocess", preprocess),
            (
                "rf",
                RandomForestClassifier(
                    class_weight=class_weight,
                    random_state=random_seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return pipe


def define_rf_grid():
    """
    Define the hyperparameter grid for Random Forest.

    Returns:
    dict: parameter search space
    """

    params = {
        "rf__n_estimators": randint(100, 500),
        "rf__max_depth": [None, 5, 10, 20, 30],
        "rf__min_samples_split": randint(2, 11),
        "rf__min_samples_leaf": randint(1, 6),
        "rf__max_features": ["sqrt", "log2", None],
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
    Tune the Random Forest model using RandomizedSearchCV.

    Parameters:
    pipe (Pipeline): pipeline containing preprocessing and Random Forest
    params (dict): hyperparameter search space
    cv: cross-validation splitter
    Xtrain: training features
    ytrain: training labels
    n_iter (int, optional): number of random search iterations
    scoring (str, optional): scoring metric for model selection
    random_seed (int, optional): random seed for reproducibility
    n_jobs (int, optional): number of parallel jobs

    Returns:
    RandomizedSearchCV: fitted search object
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
    return_train_score=True
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

    Parameters:
    preprocess: preprocessing pipeline to apply to the data
    cv: cross-validation splitter
    Xtrain: training features
    ytrain: training labels
    class_weight (str or dict, optional): class weighting strategy
    random_seed (int, optional): random seed for reproducibility
    n_iter (int, optional): number of random search iterations
    scoring (str, optional): scoring metric for model selection
    n_jobs (int, optional): number of parallel jobs

    Returns:
    tuple: fitted RandomizedSearchCV object, pipeline, parameter grid
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
    Save the trained model to the models folder.
    """
    out_dir = Path(__file__).parent / "../models"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_path = out_dir / filename
    joblib.dump(model, save_path)
    print(f"[SUCCESS] Model saved to {save_path}")