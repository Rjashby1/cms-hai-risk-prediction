import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

def run_random_forest(X_train, y_train, params=None, random_seed=42):
    """
    Fits a Random Forest model. 
    Functional for Baseline, Pruned (limited X), or Tuned (via params).
    """
    
    # Default parameters for Baseline
    # 'class_weight="balanced"' is the functional fix for the rare HAI events
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'class_weight': 'balanced',
            'n_jobs': -1 # Use all available CPU cores
        }
    
    # Initialize model with seed and parameters
    model = RandomForestClassifier(random_state=random_seed, **params)
    
    # Fit model
    print(f"[MODELING] Training Random Forest...")
    model.fit(X_train, y_train)
    
    return model

def save_model(model, filename="randforest_model.joblib"):
    """Saves the trained model to the models folder."""
    out_dir = Path(__file__).parent / "../models"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = out_dir / filename
    joblib.dump(model, save_path)
    print(f"[SUCCESS] Model saved to {save_path}")