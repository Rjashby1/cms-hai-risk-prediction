import pandas as pd
from pathlib import Path


def _standardize_facility_id(df: pd.DataFrame, col_name: str = "Facility ID") -> pd.DataFrame:
    """
    Return a copy of the DataFrame with Facility ID standardized to a
    zero-padded 6-character string.
    """
    df = df.copy()
    df[col_name] = df[col_name].astype(str).str.zfill(6)
    return df


def _identify_hai_sir_rows(hai_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only HAI SIR rows from the CMS healthcare-associated infections file.

    This uses the Measure ID field and keeps rows containing both HAI and SIR.
    """
    temp_df = hai_df.copy()
    temp_df["Measure ID"] = temp_df["Measure ID"].astype(str).str.strip()

    mask = (
        temp_df["Measure ID"].str.contains("HAI", case=False, na=False)
        & temp_df["Measure ID"].str.contains("SIR", case=False, na=False)
    )

    return temp_df.loc[mask].copy()


def build_target_and_master(df_dict, output_dir="../data/processed"):
    """
    Construct the binary HAI risk target and master modeling file.

    Target definition
    -----------------
    A facility is labeled HAI_at_risk = 1 if any reported HAI SIR measure is
    greater than 1.0, indicating worse-than-benchmark performance on at least
    one tracked healthcare-associated infection metric.

    In addition to the primary binary target, this function also saves a richer
    facility-level target table with counts of reported HAI SIR measures and
    counts of worse-than-benchmark measures.

    Parameters
    ----------
    df_dict : dict
        Dictionary of raw CMS DataFrames.
    output_dir : str, default="../data/processed"
        Relative output directory for processed target files.

    Returns
    -------
    pd.DataFrame
        Master modeling DataFrame containing:
        Facility ID, HAI_at_risk, Hospital Type, Hospital Ownership,
        Hospital overall rating
    """
    current_dir = Path(__file__).parent
    out_path = (current_dir / output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load and standardize required source files
    # ------------------------------------------------------------------
    hai_df = df_dict["Healthcare_Associated_Infections-Hospital"].copy()
    gen_df = df_dict["Hospital_General_Information"].copy()

    hai_df = _standardize_facility_id(hai_df, "Facility ID")
    gen_df = _standardize_facility_id(gen_df, "Facility ID")

    # ------------------------------------------------------------------
    # 2. Isolate HAI SIR rows only
    # ------------------------------------------------------------------
    target_prep = _identify_hai_sir_rows(hai_df)

    # Convert Score to numeric. Non-numeric entries become NaN.
    target_prep["Score_Numeric"] = pd.to_numeric(target_prep["Score"], errors="coerce")

    # Binary indicator at the row level: worse than benchmark if SIR > 1.0
    target_prep["is_worse"] = (target_prep["Score_Numeric"] > 1.0).astype(int)

    # ------------------------------------------------------------------
    # 3. Facility-level summary counts
    # ------------------------------------------------------------------
    facility_summary = (
        target_prep.groupby("Facility ID")
        .agg(
            n_hai_sir_reported=("Score_Numeric", lambda x: x.notna().sum()),
            n_hai_sir_worse=("is_worse", "sum"),
        )
        .reset_index()
    )

    facility_summary["HAI_at_risk"] = (facility_summary["n_hai_sir_worse"] > 0).astype(int)

    # ------------------------------------------------------------------
    # 4. Optional facility-level flags by HAI subtype
    # ------------------------------------------------------------------
    subtype_flags = (
        target_prep.pivot_table(
            index="Facility ID",
            columns="Measure ID",
            values="is_worse",
            aggfunc="max",
            fill_value=0,
        )
        .reset_index()
    )

    # Clean column names slightly for easier downstream use
    subtype_flags.columns.name = None

    # ------------------------------------------------------------------
    # 5. Combine target summary pieces
    # ------------------------------------------------------------------
    target_detail_df = pd.merge(
        facility_summary,
        subtype_flags,
        on="Facility ID",
        how="left"
    )

    # ------------------------------------------------------------------
    # 6. Merge with structural hospital fields for master modeling file
    # ------------------------------------------------------------------
    structural_cols = [
        "Facility ID",
        "Hospital Type",
        "Hospital Ownership",
        "Hospital overall rating",
    ]

    master_df = pd.merge(
        target_detail_df[["Facility ID", "HAI_at_risk"]],
        gen_df[structural_cols],
        on="Facility ID",
        how="inner"
    )

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    target_detail_file = out_path / "facility_hai_target_detail.csv"
    master_file = out_path / "master_model_data.csv"

    target_detail_df.to_csv(target_detail_file, index=False)
    master_df.to_csv(master_file, index=False)

    # ------------------------------------------------------------------
    # 8. Reporting
    # ------------------------------------------------------------------
    print("--- 03 Processing Complete ---")
    print(f"Total Facilities in target detail file: {len(target_detail_df)}")
    print(f"Total Facilities in master modeling file: {len(master_df)}")
    print(f"At Risk Count: {master_df['HAI_at_risk'].sum()}")
    print(f"[SUCCESS] Saved target detail file to: {target_detail_file}")
    print(f"[SUCCESS] Saved master modeling file to: {master_file}")

    return master_df


if __name__ == "__main__":
    from importlib import import_module

    step_01 = import_module("01_data_import")
    all_data = step_01.load_all_raw_data()

    if all_data:
        master_data = build_target_and_master(all_data)
        print(master_data.head().to_string(index=False))