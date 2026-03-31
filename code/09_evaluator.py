import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, Iterable, Optional

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


DEFAULT_FRIENDLY_LABELS = {
    "PSI_90": "Serious complications composite (PSI-90)",
    "PSI_03": "Pressure ulcer rate",
    "PSI_06": "Iatrogenic pneumothorax rate",
    "PSI_08": "Postoperative hip fracture rate",
    "PSI_09": "Postoperative hemorrhage or hematoma rate",
    "PSI_10": "Postoperative kidney or metabolic complication rate",
    "PSI_11": "Postoperative respiratory failure rate",
    "PSI_12": "Postoperative pulmonary embolism or DVT rate",
    "PSI_13": "Postoperative sepsis rate",
    "PSI_14": "Postoperative wound dehiscence rate",
    "PSI_15": "Accidental puncture or laceration rate",
    "Hybrid_HWM": "Hospital-wide all-cause mortality rate",
    "Hybrid_HWR": "Hospital-wide all-cause readmission rate",
    "MORT_30_COPD": "30-day mortality rate for COPD patients",
    "MORT_30_HF": "30-day mortality rate for heart failure patients",
    "MORT_30_PN": "30-day mortality rate for pneumonia patients",
    "OP_18a": "ED time before departure, all patients",
    "OP_18b": "ED time before departure, excluding transfers and psych patients",
    "OP_18c": "ED time before discharge home, psych/mental health patients",
    "OP_22": "Percent leaving ED before being seen",
    "OP_23": "Percent receiving brain scan results within 45 minutes for stroke symptoms",
    "SEP_1": "Sepsis and septic shock bundle performance",
    "SEP_SH_3HR": "Septic shock 3-hour bundle performance",
    "SEP_SH_6HR": "Septic shock 6-hour bundle performance",
    "SEV_SEP_3HR": "Severe sepsis 3-hour bundle performance",
    "SEV_SEP_6HR": "Severe sepsis 6-hour bundle performance",
    "EDAC_30_HF": "Excess days in acute care after heart failure hospitalization",
    "EDAC_30_PN": "Excess days in acute care after pneumonia hospitalization",
    "READM_30_COPD": "30-day readmission rate for COPD patients",
    "READM_30_HF": "30-day readmission rate for heart failure patients",
    "READM_30_PN": "30-day readmission rate for pneumonia patients",
    "OP_32": "Unplanned hospital visits after outpatient colonoscopy",
    "OP_36": "Ratio of unplanned hospital visits after outpatient surgery",
    "MSPB-1_x": "Medicare spending per beneficiary",
    "MSPB-1_y": "Medicare spending per beneficiary",
}


def _ensure_dir(path_like) -> Path:
    path = Path(path_like).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_output_dir(output_dir="../reports/figures") -> Path:
    current_dir = Path(__file__).parent
    return _ensure_dir(current_dir / output_dir)


def _get_table_dir(output_dir="../data/processed") -> Path:
    current_dir = Path(__file__).parent
    return _ensure_dir(current_dir / output_dir)


def _safe_predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.asarray(scores, dtype=float)
        score_min = scores.min()
        score_max = scores.max()

        if score_max == score_min:
            return np.full(shape=len(scores), fill_value=0.5, dtype=float)

        return (scores - score_min) / (score_max - score_min)

    raise AttributeError("Model must implement predict_proba() or decision_function().")


def relabel_feature(name: str, friendly_labels: Optional[dict] = None) -> str:
    """
    Convert raw pipeline feature names into more readable labels.
    """
    label_map = DEFAULT_FRIENDLY_LABELS.copy()
    if friendly_labels is not None:
        label_map.update(friendly_labels)

    if name.startswith("num__"):
        raw = name.replace("num__", "")
        return label_map.get(raw, raw)

    if name.startswith("cat__"):
        raw = name.replace("cat__", "")
        raw = raw.replace("_", " = ", 1)
        return raw

    return label_map.get(name, name)


def relabel_importance_table(
    importance_df: pd.DataFrame,
    feature_col: str = "feature",
    new_col: str = "feature_label",
    friendly_labels: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Add a readable feature-label column to an importance table.
    """
    out_df = importance_df.copy()
    out_df[new_col] = out_df[feature_col].apply(
        lambda x: relabel_feature(str(x), friendly_labels=friendly_labels)
    )
    return out_df


def evaluate_classifier(
    model,
    X_eval,
    y_eval,
    model_name: str,
    threshold: float = 0.50,
    save_confusion_plot: bool = False,
    figures_dir: str = "../reports/figures",
    normalize_confusion: Optional[str] = None,
):
    """
    Evaluate a fitted classifier on a held-out dataset.
    """
    y_true = np.asarray(y_eval)
    y_prob = _safe_predict_proba(model, X_eval)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    clf_report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

    results = {
        "model": model_name,
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, y_prob),
        "avg_precision": average_precision_score(y_true, y_prob),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "confusion_matrix": cm,
        "classification_report": clf_report,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }

    print(f"\n===== {model_name} =====")
    print(f"Threshold: {threshold:.2f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    if save_confusion_plot:
        out_dir = _get_output_dir(figures_dir)

        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, values_format="d", cmap="Blues", colorbar=False)
        ax.set_title(f"{model_name} Confusion Matrix")
        fig.tight_layout()
        fig.savefig(out_dir / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=300)
        plt.close(fig)

        if normalize_confusion is not None:
            cm_norm = confusion_matrix(y_true, y_pred, normalize=normalize_confusion)
            fig, ax = plt.subplots(figsize=(6, 5))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm)
            disp.plot(ax=ax, values_format=".2f", cmap="Blues", colorbar=False)
            ax.set_title(f"{model_name} Confusion Matrix ({normalize_confusion}-normalized)")
            fig.tight_layout()
            fig.savefig(
                out_dir / f"{model_name.lower().replace(' ', '_')}_confusion_matrix_normalized.png",
                dpi=300
            )
            plt.close(fig)

    return results


def compare_models(
    models: Dict[str, object],
    X_eval,
    y_eval,
    threshold: float = 0.50,
    save_csv: bool = False,
    csv_name: str = "heldout_model_comparison.csv",
    output_dir: str = "../data/processed",
):
    """
    Evaluate multiple fitted models on the same held-out dataset.
    """
    rows = []

    for model_name, model in models.items():
        result = evaluate_classifier(
            model=model,
            X_eval=X_eval,
            y_eval=y_eval,
            model_name=model_name,
            threshold=threshold,
            save_confusion_plot=False,
        )

        rows.append({
            "model": result["model"],
            "threshold": result["threshold"],
            "roc_auc": result["roc_auc"],
            "avg_precision": result["avg_precision"],
            "balanced_accuracy": result["balanced_accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "mcc": result["mcc"],
            "tn": result["tn"],
            "fp": result["fp"],
            "fn": result["fn"],
            "tp": result["tp"],
        })

    results_df = pd.DataFrame(rows).sort_values(by="roc_auc", ascending=False).reset_index(drop=True)

    if save_csv:
        out_dir = _get_table_dir(output_dir)
        results_df.to_csv(out_dir / csv_name, index=False)
        print(f"[SUCCESS] Saved model comparison table to {out_dir / csv_name}")

    return results_df


def threshold_sweep(
    model,
    X_eval,
    y_eval,
    model_name: str,
    thresholds: Optional[Iterable[float]] = None,
    save_csv: bool = False,
    csv_name: Optional[str] = None,
    output_dir: str = "../data/processed",
):
    """
    Evaluate one fitted model across multiple probability thresholds.
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.05)

    rows = []
    y_true = np.asarray(y_eval)
    y_prob = _safe_predict_proba(model, X_eval)

    roc_auc = roc_auc_score(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        rows.append({
            "model": model_name,
            "threshold": round(float(threshold), 4),
            "roc_auc": roc_auc,
            "avg_precision": avg_precision,
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        })

    sweep_df = pd.DataFrame(rows)

    if save_csv:
        out_dir = _get_table_dir(output_dir)
        if csv_name is None:
            csv_name = f"{model_name.lower().replace(' ', '_')}_threshold_sweep.csv"
        sweep_df.to_csv(out_dir / csv_name, index=False)
        print(f"[SUCCESS] Saved threshold sweep table to {out_dir / csv_name}")

    return sweep_df


def plot_roc_curves(
    models: Dict[str, object],
    X_eval,
    y_eval,
    save_plot: bool = False,
    filename: str = "roc_curves.png",
    figures_dir: str = "../reports/figures",
):
    """
    Plot ROC curves for multiple fitted models.
    """
    y_true = np.asarray(y_eval)

    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, model in models.items():
        y_prob = _safe_predict_proba(model, X_eval)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_title("ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save_plot:
        out_dir = _get_output_dir(figures_dir)
        fig.savefig(out_dir / filename, dpi=300)
        print(f"[SUCCESS] Saved ROC curve plot to {out_dir / filename}")

    return fig, ax


def plot_precision_recall_curves(
    models: Dict[str, object],
    X_eval,
    y_eval,
    save_plot: bool = False,
    filename: str = "precision_recall_curves.png",
    figures_dir: str = "../reports/figures",
):
    """
    Plot precision-recall curves for multiple fitted models.
    """
    y_true = np.asarray(y_eval)

    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, model in models.items():
        y_prob = _safe_predict_proba(model, X_eval)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, linewidth=2, label=f"{model_name} (AP={ap:.3f})")

    ax.set_title("Precision-Recall Curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="best")
    fig.tight_layout()

    if save_plot:
        out_dir = _get_output_dir(figures_dir)
        fig.savefig(out_dir / filename, dpi=300)
        print(f"[SUCCESS] Saved precision-recall plot to {out_dir / filename}")

    return fig, ax


def get_feature_names_from_pipeline(fitted_pipeline):
    """
    Recover transformed feature names from a fitted sklearn Pipeline
    containing a 'preprocess' step.
    """
    if not hasattr(fitted_pipeline, "named_steps"):
        raise ValueError("Expected a fitted sklearn Pipeline.")

    if "preprocess" not in fitted_pipeline.named_steps:
        raise ValueError("Pipeline does not contain a 'preprocess' step.")

    preprocess = fitted_pipeline.named_steps["preprocess"]

    if hasattr(preprocess, "get_feature_names_out"):
        names = preprocess.get_feature_names_out()
        return list(names)

    raise ValueError("Could not recover feature names from preprocessing pipeline.")


def extract_logreg_importance(
    fitted_pipeline,
    top_n: Optional[int] = 20,
    sort_by_abs: bool = True,
    save_csv: bool = False,
    csv_name: str = "logreg_feature_importance.csv",
    output_dir: str = "../data/processed",
    relabel: bool = False,
    friendly_labels: Optional[dict] = None,
):
    """
    Extract coefficient-based importance from a fitted Logistic Regression pipeline.
    """
    if "logreg" not in fitted_pipeline.named_steps:
        raise ValueError("Pipeline does not contain a 'logreg' step.")

    feature_names = get_feature_names_from_pipeline(fitted_pipeline)
    coefs = fitted_pipeline.named_steps["logreg"].coef_.ravel()

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
        "odds_ratio": np.exp(coefs),
    })

    if relabel:
        imp_df = relabel_importance_table(
            imp_df,
            feature_col="feature",
            new_col="feature_label",
            friendly_labels=friendly_labels,
        )

    if sort_by_abs:
        imp_df = imp_df.sort_values("abs_coefficient", ascending=False)
    else:
        imp_df = imp_df.sort_values("coefficient", ascending=False)

    if top_n is not None:
        imp_df = imp_df.head(top_n).reset_index(drop=True)
    else:
        imp_df = imp_df.reset_index(drop=True)

    if save_csv:
        out_dir = _get_table_dir(output_dir)
        imp_df.to_csv(out_dir / csv_name, index=False)
        print(f"[SUCCESS] Saved Logistic Regression importance table to {out_dir / csv_name}")

    return imp_df


def extract_tree_importance(
    fitted_pipeline,
    model_step_name: str,
    top_n: Optional[int] = 20,
    save_csv: bool = False,
    csv_name: Optional[str] = None,
    output_dir: str = "../data/processed",
    relabel: bool = False,
    friendly_labels: Optional[dict] = None,
):
    """
    Extract feature importance from a fitted tree-based pipeline.
    """
    if model_step_name not in fitted_pipeline.named_steps:
        raise ValueError(f"Pipeline does not contain a '{model_step_name}' step.")

    model = fitted_pipeline.named_steps[model_step_name]

    if not hasattr(model, "feature_importances_"):
        raise ValueError(f"Estimator '{model_step_name}' does not expose feature_importances_.")

    feature_names = get_feature_names_from_pipeline(fitted_pipeline)
    importances = model.feature_importances_

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })

    if relabel:
        imp_df = relabel_importance_table(
            imp_df,
            feature_col="feature",
            new_col="feature_label",
            friendly_labels=friendly_labels,
        )

    imp_df = imp_df.sort_values("importance", ascending=False)

    if top_n is not None:
        imp_df = imp_df.head(top_n).reset_index(drop=True)
    else:
        imp_df = imp_df.reset_index(drop=True)

    if save_csv:
        out_dir = _get_table_dir(output_dir)
        if csv_name is None:
            csv_name = f"{model_step_name}_feature_importance.csv"
        imp_df.to_csv(out_dir / csv_name, index=False)
        print(f"[SUCCESS] Saved {model_step_name} importance table to {out_dir / csv_name}")

    return imp_df


def extract_permutation_importance(
    fitted_pipeline,
    X_eval,
    y_eval,
    scoring: str = "roc_auc",
    n_repeats: int = 20,
    random_seed: int = 42,
    top_n: Optional[int] = 20,
    save_csv: bool = False,
    csv_name: str = "permutation_feature_importance.csv",
    output_dir: str = "../data/processed",
    relabel: bool = False,
    friendly_labels: Optional[dict] = None,
):
    """
    Extract permutation importance on an evaluation dataset.
    """
    result = permutation_importance(
        fitted_pipeline,
        X_eval,
        y_eval,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_seed,
        n_jobs=-1,
    )

    feature_names = get_feature_names_from_pipeline(fitted_pipeline)

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    if relabel:
        imp_df = relabel_importance_table(
            imp_df,
            feature_col="feature",
            new_col="feature_label",
            friendly_labels=friendly_labels,
        )

    if top_n is not None:
        imp_df = imp_df.head(top_n).reset_index(drop=True)
    else:
        imp_df = imp_df.reset_index(drop=True)

    if save_csv:
        out_dir = _get_table_dir(output_dir)
        imp_df.to_csv(out_dir / csv_name, index=False)
        print(f"[SUCCESS] Saved permutation importance table to {out_dir / csv_name}")

    return imp_df


def plot_top_features(
    importance_df: pd.DataFrame,
    feature_col: str,
    value_col: str,
    title: str,
    top_n: int = 10,
    save_plot: bool = False,
    filename: Optional[str] = None,
    figures_dir: str = "../reports/figures",
):
    """
    Plot a horizontal bar chart for top features from an importance table.
    """
    plot_df = importance_df.head(top_n).copy()
    plot_df = plot_df.sort_values(by=value_col, ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df[feature_col], plot_df[value_col])
    ax.set_title(title)
    ax.set_xlabel(value_col)
    ax.set_ylabel("Feature")
    fig.tight_layout()

    if save_plot:
        out_dir = _get_output_dir(figures_dir)
        if filename is None:
            filename = f"{title.lower().replace(' ', '_')}.png"
        fig.savefig(out_dir / filename, dpi=300)
        print(f"[SUCCESS] Saved feature importance plot to {out_dir / filename}")

    return fig, ax