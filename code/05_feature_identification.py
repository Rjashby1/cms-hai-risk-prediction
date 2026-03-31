import pandas as pd
from pathlib import Path


VALUE_COLUMN_CANDIDATES = [
    "Score",
    "Value",
    "Numeric Score",
    "HCAHPS Linear Mean Score",
]


def _standardize_facility_id(df: pd.DataFrame, col_name: str = "Facility ID") -> pd.DataFrame:
    """
    Return a copy of the DataFrame with Facility ID standardized to a
    zero-padded 6-character string.
    """
    df = df.copy()
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


def mine_features(
    df_dict,
    overlap_threshold=0.50,
    completion_threshold=0.50,
    exclude_files=None,
    input_dir="../data/processed",
    output_dir="../data/processed",
):
    """
    Automated feature discovery for facility-level CMS hospital modeling.

    This step anchors on the master modeling file created in Step 03 and then
    scans eligible CMS datasets for additional facility-level predictors.

    Inclusion logic
    ---------------
    A dataset is only considered if:
    1. it is not explicitly excluded,
    2. it contains a Facility ID column,
    3. it has sufficient overlap with the target facility cohort, and
    4. it contains both a Measure ID column and a usable value column.

    After pivoting from long to wide, individual feature columns are filtered by
    completion threshold, meaning that features with too much missingness are
    removed before the final merge.

    Important methodological note
    -----------------------------
    This function does NOT perform final imputation. Missing values are
    preserved and should be handled later inside model preprocessing pipelines.

    Parameters
    ----------
    df_dict : dict
        Dictionary of raw CMS DataFrames.
    overlap_threshold : float, default=0.50
        Minimum share of target facilities that must appear in a dataset.
    completion_threshold : float, default=0.50
        Minimum non-missing proportion required to retain an individual feature.
    exclude_files : list[str] or None
        Exact dataset names to exclude entirely from mining.
    input_dir : str, default="../data/processed"
        Directory containing master_model_data.csv.
    output_dir : str, default="../data/processed"
        Directory where the final feature set and feature manifest will be saved.

    Returns
    -------
    pd.DataFrame
        Final merged feature set anchored to the facility-level master dataset.
    """
    current_dir = Path(__file__).parent
    input_path = (current_dir / input_dir).resolve()
    output_path = (current_dir / output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    master_path = input_path / "master_model_data.csv"
    master_df = pd.read_csv(master_path)
    master_df = _standardize_facility_id(master_df, "Facility ID")

    target_ids = set(master_df["Facility ID"])
    print(f"Targeting predictors for {len(target_ids)} facilities...")

    if exclude_files is None:
        exclude_files = ["Healthcare_Associated_Infections-Hospital"]

    # Skip obvious non-facility / low-value summary files by name pattern
    skip_keywords = [
        "Footnote",
        "State",
        "National",
        "ZIP",
        "Date",
        "Crosswalk",
        "Updates",
    ]

    included_sources = []
    feature_manifest_rows = []

    for name, df in df_dict.items():
        # ------------------------------------------------------------
        # Source-level exclusion checks
        # ------------------------------------------------------------
        if name in exclude_files:
            print(f"[SKIP - EXCLUDED SOURCE] {name}")
            continue

        if any(key in name for key in skip_keywords):
            continue

        if "Facility ID" not in df.columns:
            continue

        if "Measure ID" not in df.columns:
            continue

        df = _standardize_facility_id(df, "Facility ID")

        overlap_count = len(set(df["Facility ID"]) & target_ids)
        overlap_share = overlap_count / len(target_ids)

        if overlap_share < overlap_threshold:
            continue

        value_col = _find_value_column(df)
        if value_col is None:
            continue

        # ------------------------------------------------------------
        # Value conversion and pivot
        # ------------------------------------------------------------
        temp_df = df.copy()
        temp_df[value_col] = pd.to_numeric(temp_df[value_col], errors="coerce")

        pivot_df = temp_df.pivot_table(
            index="Facility ID",
            columns="Measure ID",
            values=value_col,
            aggfunc="mean"
        ).reset_index()

        pivot_df = pivot_df[pivot_df["Facility ID"].isin(target_ids)].copy()

        # ------------------------------------------------------------
        # Column-level completion filter
        # ------------------------------------------------------------
        retained_cols = ["Facility ID"]
        for col in pivot_df.columns:
            if col == "Facility ID":
                continue

            non_missing_count = pivot_df[col].notna().sum()
            completion_rate = non_missing_count / len(target_ids)

            if completion_rate >= completion_threshold:
                retained_cols.append(col)

                feature_manifest_rows.append({
                    "feature_name": col,
                    "source_dataset": name,
                    "value_column_used": value_col,
                    "overlap_count": overlap_count,
                    "overlap_rate": round(overlap_share, 4),
                    "non_missing_count": int(non_missing_count),
                    "completion_rate": round(completion_rate, 4),
                })

        pivot_df = pivot_df[retained_cols].copy()

        if pivot_df.shape[1] > 1:
            master_df = pd.merge(master_df, pivot_df, on="Facility ID", how="left")
            included_sources.append(name)
            print(
                f"[DATA DISCOVERED] {name}: "
                f"Added {pivot_df.shape[1] - 1} features."
            )

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------
    final_feature_path = output_path / "final_modeling_feature_set.csv"
    manifest_path = output_path / "feature_manifest.csv"

    master_df.to_csv(final_feature_path, index=False)

    feature_manifest_df = pd.DataFrame(feature_manifest_rows)
    if not feature_manifest_df.empty:
        feature_manifest_df = feature_manifest_df.sort_values(
            by=["source_dataset", "completion_rate", "feature_name"],
            ascending=[True, False, True]
        ).reset_index(drop=True)
        feature_manifest_df.to_csv(manifest_path, index=False)
    else:
        feature_manifest_df.to_csv(manifest_path, index=False)

    # ------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------
    print("-" * 30)
    print(f"[SUCCESS] Final Feature Set Shape: {master_df.shape}")
    print(f"[INFO] Included feature sources: {included_sources}")
    print(f"[INFO] Excluded feature sources: {exclude_files}")
    print(f"[SUCCESS] Saved final feature set to: {final_feature_path}")
    print(f"[SUCCESS] Saved feature manifest to: {manifest_path}")
    print("[INFO] Numeric missing values were preserved for downstream")
    print("       preprocessing and training-only imputation.")

    return master_df


if __name__ == "__main__":
    from importlib import import_module

    step_01 = import_module("01_data_import")
    all_data = step_01.load_all_raw_data()

    if all_data:
        final_df = mine_features(all_data)
        print(final_df.head().to_string(index=False))