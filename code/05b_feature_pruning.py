import pandas as pd
import numpy as np
from pathlib import Path


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


def _find_correlated_features(df: pd.DataFrame, threshold: float = 0.85):
    """
    Identify numeric features to drop based on absolute pairwise correlation.

    The procedure keeps the first feature encountered and drops later features
    that exceed the correlation threshold with an earlier feature.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing numeric modeling features only.
    threshold : float, default=0.85
        Absolute correlation threshold above which a feature is dropped.

    Returns
    -------
    tuple
        (features_to_drop, correlation_detail_df)
    """
    if df.shape[1] == 0:
        empty_detail = pd.DataFrame(
            columns=["feature_kept", "feature_dropped", "abs_correlation"]
        )
        return [], empty_detail

    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = []
    detail_rows = []

    for col in upper_triangle.columns:
        high_corr_partners = upper_triangle.index[upper_triangle[col] > threshold].tolist()

        if len(high_corr_partners) > 0:
            to_drop.append(col)
            for kept_feature in high_corr_partners:
                detail_rows.append({
                    "feature_kept": kept_feature,
                    "feature_dropped": col,
                    "abs_correlation": round(float(upper_triangle.loc[kept_feature, col]), 6),
                })

    detail_df = pd.DataFrame(detail_rows).sort_values(
        by=["abs_correlation", "feature_dropped"],
        ascending=[False, True]
    ).reset_index(drop=True) if detail_rows else pd.DataFrame(
        columns=["feature_kept", "feature_dropped", "abs_correlation"]
    )

    return sorted(set(to_drop)), detail_df


def prune_features(
    input_csv="../data/processed/final_modeling_feature_set.csv",
    output_dir="../data/processed",
    target_col="HAI_at_risk",
    manual_drop_cols=None,
    correlation_threshold=0.85,
    run_correlation_pruning=True,
):
    """
    Apply feature governance and pruning to the CMS hospital modeling feature set.

    This step is intended to sit between broad feature mining and formal model
    training. It performs two main forms of pruning:

    1. Manual / methodological exclusions
       These are variables removed by design, such as identifiers, structural
       context variables, or fields judged to be non-actionable for the project’s
       hospital leadership use case.

    2. Correlation-based pruning
       Highly redundant numeric predictors are removed using an absolute
       correlation threshold.

    Parameters
    ----------
    input_csv : str, default="../data/processed/final_modeling_feature_set.csv"
        Relative path to the feature set created in Step 05.
    output_dir : str, default="../data/processed"
        Relative directory where pruned outputs will be saved.
    target_col : str, default="HAI_at_risk"
        Name of the binary modeling target.
    manual_drop_cols : list[str] or None
        Columns to drop before statistical pruning.
    correlation_threshold : float, default=0.85
        Absolute correlation threshold for dropping redundant numeric features.
    run_correlation_pruning : bool, default=True
        Whether to apply correlation-based pruning after manual exclusions.

    Returns
    -------
    dict
        Dictionary containing:
        - pruned_df
        - retained_features
        - manually_dropped_features
        - correlation_dropped_features
        - pruning_summary_df
        - correlation_detail_df
    """
    current_dir = Path(__file__).parent
    input_path = (current_dir / input_csv).resolve()
    output_path = _ensure_dir(current_dir / output_dir)

    df = pd.read_csv(input_path)
    df = _standardize_facility_id(df, "Facility ID")

    if manual_drop_cols is None:
        manual_drop_cols = [
            "Facility ID",
            "Hospital overall rating",
            "Hospital Type",
            "Hospital Ownership",
        ]

    manual_drop_cols_present = [col for col in manual_drop_cols if col in df.columns]

    # ------------------------------------------------------------
    # 1. Manual / methodological exclusions
    # ------------------------------------------------------------
    pruned_df = df.drop(columns=manual_drop_cols_present, errors="ignore").copy()

    # ------------------------------------------------------------
    # 2. Correlation-based pruning on numeric predictors only
    # ------------------------------------------------------------
    correlation_dropped_features = []
    correlation_detail_df = pd.DataFrame(
        columns=["feature_kept", "feature_dropped", "abs_correlation"]
    )

    if run_correlation_pruning:
        numeric_cols = pruned_df.select_dtypes(include=["number"]).columns.tolist()
        numeric_predictors = [col for col in numeric_cols if col != target_col]

        corr_input_df = pruned_df[numeric_predictors].copy()
        correlation_dropped_features, correlation_detail_df = _find_correlated_features(
            corr_input_df,
            threshold=correlation_threshold
        )

        if len(correlation_dropped_features) > 0:
            pruned_df = pruned_df.drop(columns=correlation_dropped_features, errors="ignore")

    # ------------------------------------------------------------
    # 3. Retained features list
    # ------------------------------------------------------------
    retained_features = [col for col in pruned_df.columns if col != target_col]

    # ------------------------------------------------------------
    # 4. Summary report
    # ------------------------------------------------------------
    original_feature_count = len([col for col in df.columns if col != target_col])
    final_feature_count = len(retained_features)

    pruning_summary_df = pd.DataFrame([
        {
            "original_feature_count": original_feature_count,
            "manual_drop_count": len(manual_drop_cols_present),
            "correlation_drop_count": len(correlation_dropped_features),
            "final_feature_count": final_feature_count,
            "correlation_threshold": correlation_threshold if run_correlation_pruning else None,
        }
    ])

    manually_dropped_df = pd.DataFrame({
        "manually_dropped_feature": manual_drop_cols_present
    })

    correlation_dropped_df = pd.DataFrame({
        "correlation_dropped_feature": correlation_dropped_features
    })

    retained_features_df = pd.DataFrame({
        "retained_feature": retained_features
    })

    # ------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------
    pruned_feature_path = output_path / "pruned_modeling_feature_set.csv"
    pruning_summary_path = output_path / "feature_pruning_summary.csv"
    manual_drop_path = output_path / "manual_dropped_features.csv"
    correlation_drop_path = output_path / "correlation_dropped_features.csv"
    correlation_detail_path = output_path / "correlation_pruning_detail.csv"
    retained_features_path = output_path / "retained_model_features.csv"

    pruned_df.to_csv(pruned_feature_path, index=False)
    pruning_summary_df.to_csv(pruning_summary_path, index=False)
    manually_dropped_df.to_csv(manual_drop_path, index=False)
    correlation_dropped_df.to_csv(correlation_drop_path, index=False)
    correlation_detail_df.to_csv(correlation_detail_path, index=False)
    retained_features_df.to_csv(retained_features_path, index=False)

    # ------------------------------------------------------------
    # 6. Reporting
    # ------------------------------------------------------------
    print("[SUCCESS] Feature governance and pruning complete.")
    print(f"[INFO] Original feature count: {original_feature_count}")
    print(f"[INFO] Manual drop count: {len(manual_drop_cols_present)}")
    print(f"[INFO] Correlation drop count: {len(correlation_dropped_features)}")
    print(f"[INFO] Final feature count: {final_feature_count}")
    print(f"[SUCCESS] Saved pruned feature set to: {pruned_feature_path}")
    print(f"[SUCCESS] Saved pruning summary to: {pruning_summary_path}")
    print(f"[SUCCESS] Saved retained features to: {retained_features_path}")

    return {
        "pruned_df": pruned_df,
        "retained_features": retained_features,
        "manually_dropped_features": manual_drop_cols_present,
        "correlation_dropped_features": correlation_dropped_features,
        "pruning_summary_df": pruning_summary_df,
        "correlation_detail_df": correlation_detail_df,
    }


if __name__ == "__main__":
    results = prune_features()
    print(results["pruning_summary_df"].to_string(index=False))