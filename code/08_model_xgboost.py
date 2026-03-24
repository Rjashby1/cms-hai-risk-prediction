import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from scipy.stats import loguniform, randint

def build_xgb_pipeline(preprocess, scale_pos_weight=None, random_seed=42):
    '''
    Builds our XGBoost model.

    Parameters:
    preprocess (Pipeline): Preprocessing pipeline to apply to the data.
    scale_pos_weight (float, optional): Balancing of positive and negative weights. Default is None.
    random_seed (int, optional): Random seed for reproducibility.
    '''
    
    # Initialize XGBoost pipeline
    pipe = Pipeline([('preprocess', preprocess),
                     ('xgb', XGBClassifier(scale_pos_weight=scale_pos_weight, 
                                           eval_metric='logloss',
                                           random_state=random_seed))
    ])
    
    return pipe


def define_xgb_grid():
    '''Defines the hyperparameter grid for XGBoost.'''
    
    params = {
        'xgb__n_estimators': randint(100, 500),
        'xgb__learning_rate': loguniform(0.01, 0.3),
        'xgb__max_depth': randint(3, 10),
        'xgb__min_child_weight': randint(1, 6),
        'xgb__subsample': np.linspace(0.6, 1.0, 5),
        'xgb__colsample_bytree': np.linspace(0.6, 1.0, 5)
    }
    
    return params


def tune_xgb_model(pipe, params, cv, Xtrain, ytrain, n_iter=50, scoring='accuracy', random_seed=42, n_jobs=-1):
    '''
    Tuning the XGBoost model using RandomizedSearchCV.
    
    Parameters:
    pipe (Pipeline): The pipeline containing the XGBoost model.
    params (dict): The hyperparameter grid for tuning.
    cv (StratifiedKFold): Cross-validation strategy.
    Xtrain (array-like): Training features.
    ytrain (array-like): Training labels.
    n_iter (int, optional): Number of iterations for RandomizedSearchCV. Default is 50.
    scoring (str, optional): Scoring metric for RandomizedSearchCV. Default is 'accuracy'.
    random_seed (int, optional): Random seed for reproducibility. Default is 42.
    n_jobs (int, optional): Number of parallel jobs for RandomizedSearchCV. Default is -1.

    Returns:
    RandomizedSearchCV: The fitted RandomizedSearchCV object containing the best model and results.
    '''
    
    xgb_cv = RandomizedSearchCV(estimator=pipe, param_distributions=params, n_iter=n_iter, 
                                      cv=cv, scoring=scoring, random_state=random_seed, n_jobs=n_jobs)
    xgb_cv.fit(Xtrain, ytrain)
    
    return xgb_cv

def run_xgb_model(preprocess, cv, Xtrain, ytrain, scale_pos_weight=None, random_seed=42, n_iter=50, scoring='accuracy', n_jobs=-1):
    '''
    Runs the entire process of building, tuning, and fitting our XGBoost model.

    Parameters:
    preprocess (Pipeline): Preprocessing pipeline to apply to the data.
    cv (StratifiedKFold): Cross-validation strategy.
    Xtrain (array-like): Training features.
    ytrain (array-like): Training labels.
    scale_pos_weight (float, optional): Balancing of positive and negative weights. Default is None.
    random_seed (int, optional): Random seed for reproducibility. Default is 42.
    n_iter (int, optional): Number of iterations for RandomizedSearchCV. Default is 50.
    scoring (str, optional): Scoring metric for RandomizedSearchCV. Default is 'accuracy'.
    n_jobs (int, optional): Number of parallel jobs for RandomizedSearchCV. Default is -1.

    Returns:
    RandomizedSearchCV: The fitted RandomizedSearchCV object containing the best model and results.
    '''
    
    pipe = build_xgb_pipeline(preprocess, scale_pos_weight=scale_pos_weight, random_seed=random_seed)
    
    params = define_xgb_grid()
    
    xgb_cv = tune_xgb_model(pipe, params, cv, Xtrain, ytrain, n_iter=n_iter, scoring=scoring,
                                  random_seed=random_seed, n_jobs=n_jobs)
    
    return xgb_cv, pipe, params
