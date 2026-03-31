import numpy as np
import pandas as pd
from pathlib import Path


VALUE_COLUMN_CANDIDATES = [
    "Score",
    "Value",
    "Numeric Score",
    "HCAHPS Linear Mean Score",
]


DEFAULT_IDENTITY_COLUMNS = [
    "Facility ID",
    "Facility Name",
    "Address",
    "City/Town",
    "State",
    "ZIP Code",
    "County/Parish",
    "Telephone Number",
    "Hospital Type",
    "Hospital Ownership",
    "Emergency Services",
    "Hospital overall rating",
]


def _ensure_dir(path_like) -> Path:
    path = Path(path_like).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _standardize_facility_id(df: pd.DataFrame, col_name: str = "Facility ID") -> pd.DataFrame:
    """
    Return a copy of the DataFrame with Facility ID standardized to a
    zero-padded 6-character string.
    """
    df = df.copy()
    if col_name in df.columns:
        df[col_name] = df[col_name].astype(str).str.zfill(6)
    return df


def _find_value_column(df: pd.DataFrame):
    """
    Return the first matching candidate value column, if present.
    """
    for col in VALUE_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    return None


def _safe_predict_proba(model, X):
    """
    Return positive-class probabilities from a fitted model or pipeline.
    """
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


def _build_facility_base(
    df_dict,
    identity_columns=None,
):
    """
    Build the all-facility application base from Hospital_General_Information.
    """
    if identity_columns is None:
        identity_columns = DEFAULT_IDENTITY_COLUMNS

    if "Hospital_General_Information" not in df_dict:
        raise KeyError("Hospital_General_Information dataset not found in df_dict.")

    gen_df = df_dict["Hospital_General_Information"].copy()
    gen_df = _standardize_facility_id(gen_df, "Facility ID")

    available_identity_cols = [col for col in identity_columns if col in gen_df.columns]
    base_df = gen_df[available_identity_cols].copy()

    return base_df


def _mine_application_features(
    df_dict,
    facility_ids,
    source_names,
    exclude_files=None,
):
    """
    Rebuild the model feature space for all facilities.

    Important design note:
    This does NOT re-apply the training-time sparsity filter. Instead, it
    reconstructs a broad application feature table from the same hospital-level
    source datasets used in feature mining, then the retained feature list is
    applied afterward to align with the trained model input space.
    """
    if exclude_files is None:
        exclude_files = ["Healthcare_Associated_Infections-Hospital"]

    skip_keywords = [
        "Footnote",
        "State",
        "National",
        "ZIP",
        "Date",
        "Crosswalk",
        "Updates",
    ]

    app_feature_df = pd.DataFrame({"Facility ID": sorted(facility_ids)})

    for name, df in df_dict.items():
        if name in exclude_files:
            continue

        if name not in source_names:
            continue

        if any(key in name for key in skip_keywords):
            continue

        if "Facility ID" not in df.columns:
            continue

        if "Measure ID" not in df.columns:
            continue

        value_col = _find_value_column(df)
        if value_col is None:
            continue

        temp_df = df.copy()
        temp_df = _standardize_facility_id(temp_df, "Facility ID")
        temp_df = temp_df[temp_df["Facility ID"].isin(facility_ids)].copy()
        temp_df[value_col] = pd.to_numeric(temp_df[value_col], errors="coerce")

        pivot_df = temp_df.pivot_table(
            index="Facility ID",
            columns="Measure ID",
            values=value_col,
            aggfunc="mean",
        ).reset_index()

        if pivot_df.shape[1] > 1:
            app_feature_df = pd.merge(app_feature_df, pivot_df, on="Facility ID", how="left")
            print(f"[APPLICATION FEATURES] {name}: merged {pivot_df.shape[1] - 1} feature columns.")

    return app_feature_df


def build_application_dataset(
    df_dict,
    retained_features_path="../data/processed/retained_model_features.csv",
    feature_manifest_path="../data/processed/feature_manifest.csv",
    output_dir="../data/processed",
    identity_columns=None,
    save_csv=True,
):
    """
    Build the all-facility model application dataset.

    This function:
    1. anchors on all facilities in Hospital_General_Information,
    2. rebuilds the broad CMS feature space from the original hospital-level
       source datasets used in feature mining,
    3. aligns the columns to the final retained model feature list from Step 05b,
    4. returns a dataset ready for model application.

    Returns
    -------
    pd.DataFrame
        Application-ready DataFrame containing identity fields plus aligned
        model feature columns.
    """
    current_dir = Path(__file__).parent
    output_path = _ensure_dir(current_dir / output_dir)

    retained_path = (current_dir / retained_features_path).resolve()
    manifest_path = (current_dir / feature_manifest_path).resolve()

    retained_features_df = pd.read_csv(retained_path)
    feature_manifest_df = pd.read_csv(manifest_path)

    retained_features = retained_features_df["retained_feature"].dropna().astype(str).tolist()
    source_names = (
        feature_manifest_df["source_dataset"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    base_df = _build_facility_base(
        df_dict=df_dict,
        identity_columns=identity_columns,
    )

    facility_ids = set(base_df["Facility ID"])

    app_features_df = _mine_application_features(
        df_dict=df_dict,
        facility_ids=facility_ids,
        source_names=source_names,
        exclude_files=["Healthcare_Associated_Infections-Hospital"],
    )

    application_df = pd.merge(base_df, app_features_df, on="Facility ID", how="left")

    for col in retained_features:
        if col not in application_df.columns:
            application_df[col] = np.nan

    ordered_cols = [col for col in base_df.columns] + retained_features
    application_df = application_df[ordered_cols].copy()

    if save_csv:
        save_path = output_path / "all_facility_application_dataset.csv"
        application_df.to_csv(save_path, index=False)
        print(f"[SUCCESS] Saved application dataset to: {save_path}")

    return application_df


def apply_model_to_all_facilities(
    fitted_model,
    df_dict,
    model_name="Random Forest",
    retained_features_path="../data/processed/retained_model_features.csv",
    feature_manifest_path="../data/processed/feature_manifest.csv",
    output_dir="../data/processed",
    identity_columns=None,
    min_available_count=5,
    min_available_fraction=0.30,
    threshold=0.40,
    positive_label="Yes",
    negative_label="No",
    insufficient_label="Too little data available",
    save_csv=True,
):
    """
    Apply a fitted model to all facilities in the CMS general information file.

    Facilities with insufficient usable feature coverage are not scored and are
    assigned a status label instead.

    Parameters
    ----------
    fitted_model : sklearn estimator or pipeline
        Fully fitted model or pipeline, for example rf_best or xgb_best.
    df_dict : dict
        Dictionary of raw CMS DataFrames.
    model_name : str, default="Random Forest"
        Model name to embed in the output metadata.
    retained_features_path : str
        Path to retained model feature list from Step 05b.
    feature_manifest_path : str
        Path to feature manifest from Step 05.
    output_dir : str
        Directory where outputs will be saved.
    identity_columns : list[str] or None
        Identity columns to keep in the output.
    min_available_count : int, default=5
        Minimum non-missing retained feature count required to score a facility.
    min_available_fraction : float, default=0.30
        Minimum fraction of retained features available to score a facility.
    threshold : float, default=0.40
        Probability threshold for converting predicted probabilities into a
        binary HAI risk label.
    positive_label : str, default="Yes"
        Positive prediction label.
    negative_label : str, default="No"
        Negative prediction label.
    insufficient_label : str, default="Too little data available"
        Output label for facilities not scored.
    save_csv : bool, default=True
        Whether to save the final application output CSV.

    Returns
    -------
    pd.DataFrame
        Tableau-ready facility-level output with identities, prediction status,
        HAI risk predictions, risk percentile, and risk tier.
    """
    current_dir = Path(__file__).parent
    output_path = _ensure_dir(current_dir / output_dir)

    retained_path = (current_dir / retained_features_path).resolve()
    retained_features_df = pd.read_csv(retained_path)
    retained_features = retained_features_df["retained_feature"].dropna().astype(str).tolist()

    application_df = build_application_dataset(
        df_dict=df_dict,
        retained_features_path=retained_features_path,
        feature_manifest_path=feature_manifest_path,
        output_dir=output_dir,
        identity_columns=identity_columns,
        save_csv=True,
    )

    X_apply = application_df[retained_features].copy()

    available_feature_count = X_apply.notna().sum(axis=1)
    total_feature_count = len(retained_features)

    if total_feature_count == 0:
        raise ValueError("No retained model features were found.")

    available_feature_fraction = available_feature_count / total_feature_count

    eligible_mask = (
        (available_feature_count >= min_available_count)
        & (available_feature_fraction >= min_available_fraction)
    )

    output_df = application_df[[col for col in application_df.columns if col not in retained_features]].copy()

    output_df["model_used"] = model_name
    output_df["available_feature_count"] = available_feature_count
    output_df["available_feature_fraction"] = available_feature_fraction.round(4)
    output_df["prediction_status"] = np.where(
        eligible_mask,
        "Predicted",
        insufficient_label,
    )

    output_df["HAI_at_risk_pred_binary"] = pd.Series(pd.NA, index=output_df.index, dtype="Int64")
    output_df["HAI_at_risk_pred_label"] = insufficient_label
    output_df["HAI_at_risk_pred_probability"] = np.nan
    output_df["risk_percentile"] = np.nan
    output_df["risk_tier"] = "Not Scored"

    if eligible_mask.any():
        X_score = X_apply.loc[eligible_mask, retained_features].copy()
        y_prob = _safe_predict_proba(fitted_model, X_score)
        y_pred = (y_prob >= threshold).astype(int)

        output_df.loc[eligible_mask, "HAI_at_risk_pred_binary"] = y_pred
        output_df.loc[eligible_mask, "HAI_at_risk_pred_label"] = np.where(
            y_pred == 1,
            positive_label,
            negative_label,
        )
        output_df.loc[eligible_mask, "HAI_at_risk_pred_probability"] = y_prob

        # Relative risk ranking among predicted facilities only
        output_df.loc[eligible_mask, "risk_percentile"] = (
            output_df.loc[eligible_mask, "HAI_at_risk_pred_probability"]
            .rank(pct=True)
        )

        # Low / Medium / High risk tiers among predicted facilities only
        output_df.loc[eligible_mask, "risk_tier"] = pd.cut(
            output_df.loc[eligible_mask, "risk_percentile"],
            bins=[0.0, 0.33, 0.66, 1.0],
            labels=["Low", "Medium", "High"],
            include_lowest=True,
        )

    if save_csv:
        save_path = output_path / "all_cms_facility_hai_risk_predictions.csv"
        output_df.to_csv(save_path, index=False)
        print(f"[SUCCESS] Saved all-facility prediction output to: {save_path}")

        summary_path = output_path / "all_cms_facility_prediction_status_counts.csv"
        output_df["prediction_status"].value_counts(dropna=False).rename_axis("prediction_status").reset_index(
            name="facility_count"
        ).to_csv(summary_path, index=False)
        print(f"[SUCCESS] Saved prediction status summary to: {summary_path}")

        risk_tier_summary_path = output_path / "all_cms_facility_risk_tier_counts.csv"
        output_df["risk_tier"].value_counts(dropna=False).rename_axis("risk_tier").reset_index(
            name="facility_count"
        ).to_csv(risk_tier_summary_path, index=False)
        print(f"[SUCCESS] Saved risk tier summary to: {risk_tier_summary_path}")

    print("[INFO] Model application complete.")
    print(f"[INFO] Total facilities: {len(output_df)}")
    print(f"[INFO] Eligible for prediction: {int(eligible_mask.sum())}")
    print(f"[INFO] Not scored: {int((~eligible_mask).sum())}")

    return output_df


if __name__ == "__main__":
    print("This module is intended to be called from the notebook with a fitted model and raw_datasets.")