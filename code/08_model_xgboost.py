import numpy as np
import pandas as pd
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
    matthews_corrcoef,
    roc_curve
)
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_SEED = 42

def find_best_threshold(y_true, y_prob):
    """
    Search predicted-probability thresholds and return the one that maximises
    the F1 score.  Uses the precision–recall curve so the search is efficient
    and covers every unique predicted probability.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

   
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx])



def _ensure_plot_dir():
    try:
        base_dir = Path(__file__).parent
    except NameError:
        base_dir = Path.cwd()
    
    plot_dir = base_dir / "models" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir

def plot_metrics(y_true, y_prob, y_pred, model, feature_names=None):
    """Generate and save standard classification plots."""
    out_dir = _ensure_plot_dir()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / 'roc_curve.png', dpi=300)
    plt.close()
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_dir / 'pr_curve.png', dpi=300)
    plt.close()
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(out_dir / 'confusion_matrix.png', dpi=300)
    plt.close()


    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 8))
        
        if feature_names is not None:
            model.get_booster().feature_names = list(feature_names)
        
        plot_importance(model, max_num_features=20, importance_type='gain', 
                        title='XGBoost Feature Importance (Top 20)', 
                        xlabel='Gain', grid=False)
        plt.tight_layout()
        plt.savefig(out_dir / 'feature_importance.png', dpi=300)
        plt.close()
        

def run_xgboost(X_train, y_train, X_test, y_test, params=None, random_seed=RANDOM_SEED):
    """
    Train and evaluate an XGBoost classifier on a single train/test split.

    Supports:
    - Baseline (default params)
    - Tuned models (pass custom `params` dict)
    - Imbalanced classification (auto `scale_pos_weight`)
    """

    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    scale_pos_weight = neg / pos if pos > 0 else 1

    # Default baseline parameters
    if params is None:
        params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "scale_pos_weight": scale_pos_weight,
            "n_jobs": -1,
        }

    # Initating model
    model = XGBClassifier(random_state=random_seed, **params)

    print("[MODELING] Training XGBoost...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    f1  = f1_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)

    print("\n[RESULTS]")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {roc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print("  Confusion Matrix:")
    print(f"  {cm}")

    # Generating plots for the single split
    print("\n[PLOTTING] Generating and saving plots...")
    plot_metrics(y_test, y_prob, y_pred, model, feature_names=X_train.columns)
    print(f"  Plots saved to: {_ensure_plot_dir()}")

    return model, {
        "accuracy": acc,
        "roc_auc": roc,
        "f1": f1,
        "confusion_matrix": cm,
    }


#Stratified K-Fold cross-validation  
def run_xgb_cv(X, y, n_splits=5, random_seed=RANDOM_SEED):
    """
    5-fold stratified cross-validation with:
    - StandardScaler → XGBClassifier pipeline
    - Automatic scale_pos_weight for class imbalance
    - Threshold tuning per fold (maximise F1)
    - Rich per-fold + summary metrics
    - Generates and saves evaluation plots for the entire CV pool
    """

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    pos = np.sum(y == 1)
    neg = np.sum(y == 0)

    results = []
    
    # Storing all out of foldo predictions to plot one combined ROC/PR curve
    oof_y_true = []
    oof_y_prob = []
    oof_y_pred = []
    last_model = None  

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'='*40}")
        print(f"[FOLD {fold}]")
        print(f"{'='*40}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=neg / pos if pos > 0 else 1,
            eval_metric="logloss",
            random_state=random_seed,
            n_jobs=-1,
        )

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])

        pipeline.fit(X_train, y_train)

        # Predicted probabilities
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Threshold tuning
        threshold = find_best_threshold(y_test, y_prob)
        y_pred = (y_prob >= threshold).astype(int)
        
        # Storing combined plots
        oof_y_true.extend(y_test)
        oof_y_prob.extend(y_prob)
        oof_y_pred.extend(y_pred)
        last_model = model

        # Metrics
        roc = roc_auc_score(y_test, y_prob)

        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(rec, prec)

        f1_macro    = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        mcc         = matthews_corrcoef(y_test, y_pred)
        cm          = confusion_matrix(y_test, y_pred)

        print(f"  ROC-AUC      : {roc:.4f}")
        print(f"  PR-AUC       : {pr_auc:.4f}")
        print(f"  F1 (macro)   : {f1_macro:.4f}")
        print(f"  F1 (weighted): {f1_weighted:.4f}")
        print(f"  MCC          : {mcc:.4f}")
        print(f"  Threshold    : {threshold:.4f}")
        print(f"  Confusion Matrix:\n  {cm}")

        results.append({
            "roc_auc": roc,
            "pr_auc": pr_auc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "mcc": mcc,
            "threshold": threshold,
        })

    results_df = pd.DataFrame(results)

    print(f"\n{'='*40}")
    print("=== CV SUMMARY (MEAN) ===")
    print(f"{'='*40}")
    print(results_df.mean().to_string())
    print("\n=== CV SUMMARY (STD) ===")
    print(results_df.std().to_string())
    
    # Generate combined out of fold plots
    print("\n[PLOTTING] Generating CV combined plots...")
    plot_metrics(np.array(oof_y_true), np.array(oof_y_prob), np.array(oof_y_pred), last_model, feature_names=X.columns)
    print(f"  Plots saved to: {_ensure_plot_dir()}")

    return results_df

def save_model(model, filename="xgboost_model.joblib"):
    """Save a trained model to the models/ folder."""

    try:
        base_dir = Path(__file__).parent
    except NameError:
        base_dir = Path.cwd()

    out_dir = base_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_path = out_dir / filename
    joblib.dump(model, save_path)

    print(f"[SUCCESS] Model saved to {save_path}")
