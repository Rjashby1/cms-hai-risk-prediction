import pandas as pd
import numpy as np
from pathlib import Path

def mine_features(
    df_dict,
    overlap_threshold=0.50,
    completion_threshold=0.50,
    exclude_files=None
):
    """
    Automated feature discovery: scans eligible datasets and selects predictors
    based on population overlap and data density.

    Parameters
    ----------
    df_dict : dict
        Dictionary of raw CMS DataFrames.
    overlap_threshold : float, default=0.50
        Minimum share of target facilities that must appear in a dataset.
    completion_threshold : float, default=0.50
        Minimum non-missing proportion required to retain a feature column.
    exclude_files : list[str] or None
        Exact dataset names to exclude from feature mining.
        By default, excludes the HAI source file used to build the target.
    """
    current_dir = Path(__file__).parent
    master_path = (current_dir / "../data/processed/master_model_data.csv").resolve()
    master_df = pd.read_csv(master_path)
    master_df["Facility ID"] = master_df["Facility ID"].astype(str).str.zfill(6)

    target_ids = set(master_df["Facility ID"])
    print(f"Targeting predictors for {len(target_ids)} facilities...")

    # Files to exclude entirely from feature mining
    if exclude_files is None:
        exclude_files = ["Healthcare_Associated_Infections-Hospital"]

    # Exclude known non-hospital summary files to save CPU cycles
    skip_keywords = ["Footnote", "State", "National", "ZIP", "Date", "Crosswalk"]

    included_sources = []

    for name, df in df_dict.items():
        if name in exclude_files:
            print(f"[SKIP - EXCLUDED SOURCE] {name}")
            continue

        if any(key in name for key in skip_keywords):
            continue

        if "Facility ID" not in df.columns:
            continue

        df = df.copy()
        df["Facility ID"] = df["Facility ID"].astype(str).str.zfill(6)

        overlap_count = len(set(df["Facility ID"]) & target_ids)
        if (overlap_count / len(target_ids)) < overlap_threshold:
            continue

        try:
            val_col = next(
                (c for c in ["Score", "Value", "Numeric Score", "HCAHPS Linear Mean Score"] if c in df.columns),
                None
            )
            if not val_col:
                continue

            if "Measure ID" not in df.columns:
                continue

            temp_df = df.copy()
            temp_df[val_col] = pd.to_numeric(temp_df[val_col], errors="coerce")

            pivot_df = temp_df.pivot_table(
                index="Facility ID",
                columns="Measure ID",
                values=val_col,
                aggfunc="mean"
            ).reset_index()

            # Evaluate density after alignment to the target cohort
            pivot_df = pivot_df[pivot_df["Facility ID"].isin(target_ids)].copy()
            limit = len(target_ids) * completion_threshold
            pivot_df = pivot_df.dropna(axis=1, thresh=limit)

            if pivot_df.shape[1] > 1:
                master_df = pd.merge(master_df, pivot_df, on="Facility ID", how="left")
                included_sources.append(name)
                print(f"[DATA DISCOVERED] {name}: Added {pivot_df.shape[1] - 1} features.")

        except Exception as e:
            print(f"[FAIL] {name}: {e}")

    # Keep this for now if you want the notebook to stay runnable,
    # though later I would move imputation into preprocessing.
    numeric_cols = master_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != "HAI_at_risk"]
    master_df[numeric_cols] = master_df[numeric_cols].fillna(master_df[numeric_cols].median())

    final_path = (current_dir / "../data/processed/final_modeling_feature_set.csv").resolve()
    master_df.to_csv(final_path, index=False)

    print("-" * 30)
    print(f"[SUCCESS] Final Feature Set Shape: {master_df.shape}")
    print(f"[INFO] Included feature sources: {included_sources}")
    print(f"[INFO] Excluded feature sources: {exclude_files}")

    return master_df