import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import  RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform

def build_logreg_pipeline(preprocess, penalty='elasticnet', class_weight=None, random_seed=42):
    '''
    Builds and a Logistic Regression model.

    Parameters:
    preprocess (Pipeline): Preprocessing pipeline to apply to the data.
    penalty (str, optional): Penalty term for Logistic Regression. Default is 'elasticnet'.
    class_weight (dict, optional): Class weights for Logistic Regression. Default is None.
    random_seed (int, optional): Random seed for reproducibility. Default is 42.
    '''
    
    # Pick correct solver for Logistic Regression based on penalty
    if penalty == 'elasticnet':
        pipe = Pipeline([('preprocess', preprocess),
                     ('logreg', LogisticRegression(penalty=penalty, solver = 'saga', l1_ratio=0.5,class_weight=class_weight, 
                                                max_iter = 10000, random_state=random_seed))
    ])
    
    elif penalty == 'l1':
        pipe = Pipeline([('preprocess', preprocess),
                     ('logreg', LogisticRegression(penalty=penalty, solver = 'liblinear', class_weight=class_weight, 
                                                max_iter = 10000, random_state=random_seed))
    ])
    
    elif penalty == 'l2':
        pipe = Pipeline([('preprocess', preprocess),
                     ('logreg', LogisticRegression(penalty=penalty, solver = 'lbfgs', class_weight=class_weight, 
                                                max_iter = 10000, random_state=random_seed))
    ])
    else:
        raise ValueError("Invalid penalty. Choose from 'elasticnet', 'l1', or 'l2'.")
    
    return pipe

def define_logreg_grid(penalty):

    '''Defines the hyperparameter grid for Logistic Regression based on the specified penalty.'''
    
    if penalty == 'elasticnet':
        params = {'logreg__C': loguniform(1e-4, 1e4),
                     'logreg__l1_ratio': np.linspace(0, 1, 10)}
    elif penalty in ['l1', 'l2']:
        params = {'logreg__C': loguniform(1e-4, 1e4)}
    else:
        raise ValueError("Invalid penalty. Choose from 'elasticnet', 'l1', or 'l2'.")
    
    return params

def tune_logreg_model(pipe, params, cv, Xtrain, ytrain, n_iter=50, scoring='accuracy',random_seed=42, n_jobs=-1):
    '''
    Tunes the Logistic Regression model using RandomizedSearchCV.
    
    Parameters:
    pipe (Pipeline): The pipeline containing the Logistic Regression model.
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
    
    logreg_cv = RandomizedSearchCV(estimator=pipe, param_distributions=params, n_iter=n_iter, 
                                      cv=cv, scoring=scoring, random_state=random_seed, n_jobs=n_jobs)
    logreg_cv.fit(Xtrain, ytrain)
    
    return logreg_cv

def run_logreg_model(preprocess, cv, Xtrain, ytrain, penalty='elasticnet', class_weight=None, random_seed=42, n_iter=50, scoring='accuracy', n_jobs=-1):
    '''
    Runs the entire process of building, tuning, and fitting a Logistic Regression model.

    Parameters:
    preprocess (Pipeline): Preprocessing pipeline to apply to the data.
    cv (StratifiedKFold): Cross-validation strategy.
    Xtrain (array-like): Training features.
    ytrain (array-like): Training labels.
    penalty (str, optional): Penalty term for Logistic Regression. Default is 'elasticnet'.
    class_weight (dict, optional): Class weights for Logistic Regression. Default is None.
    random_seed (int, optional): Random seed for reproducibility. Default is 42.
    n_iter (int, optional): Number of iterations for RandomizedSearchCV. Default is 50.
    scoring (str, optional): Scoring metric for RandomizedSearchCV. Default is 'accuracy'.
    n_jobs (int, optional): Number of parallel jobs for RandomizedSearchCV. Default is -1.

    Returns:
    RandomizedSearchCV: The fitted RandomizedSearchCV object containing the best model and results.
    '''
    
    pipe = build_logreg_pipeline(preprocess, penalty=penalty, class_weight=class_weight, random_seed=random_seed)
    
    params = define_logreg_grid(penalty)
    
    logreg_cv = tune_logreg_model(pipe, params, cv, Xtrain, ytrain, n_iter=n_iter, scoring=scoring,
                                  random_seed=random_seed, n_jobs=n_jobs)
    
    return logreg_cv, pipe, params