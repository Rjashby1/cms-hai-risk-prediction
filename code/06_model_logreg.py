import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV,StratifiedKFold, TunedThresholdClassifierCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeine import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score, precision_recall_curve

def run_baseline_logreg(data, target, features):
    X = data[features]
    y = data[target]

    # Split the data into training and testing sets
    Xtrain, Xtest, ytrain,ytest = train_test_split(X, y, test_size=0.2, random_state=5, stratify=y)

    # Specify cross-validation strategy
    cv = StratifiedKFold(n_splits=5, random_state=5, shuffle=True)

    # Build preprecessing for pipeline