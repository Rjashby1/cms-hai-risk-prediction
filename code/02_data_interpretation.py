import pandas as pd
from pathlib import Path


def _find_facility_id_column(columns):
    """
    Identify the most likely facility identifier column.
    """
    normalized = {str(col).strip().lower(): col for col in columns}

    for candidate in ["facility id", "provider id"]:
        if candidate in normalized:
            return normalized[candidate]

    return None


def _find_measure_id_column(columns):
    """
    Identify whether the dataset contains a Measure ID column.
    """
    normalized = {str(col).strip().lower(): col for col in columns}

    for candidate in ["measure id"]:
        if candidate in normalized:
            return normalized[candidate]

    return None


def _find_likely_value_column(columns):
    """
    Identify the most likely numeric value column used in downstream pivoting.
    """
    normalized = {str(col).strip().lower(): col for col in columns}

    candidates = [
        "score",
        "value",
        "numeric score",
        "hcahps linear mean score",
    ]

    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    return None


def generate_data_profile(df_dict, output_dir="../data/interim"):
    """
    Analyze all loaded DataFrames and export a dataset inventory report.

    The resulting profile is designed to help identify which CMS files are
    likely to be useful for facility-level modeling, especially for feature
    mining in later steps.

    Parameters
    ----------
    df_dict : dict
        Dictionary of loaded pandas DataFrames.
    output_dir : str, default="../data/interim"
        Relative path to the interim output directory.

    Returns
    -------
    pd.DataFrame
        Dataset-level profile table.
    """
    current_dir = Path(__file__).parent
    out_path = (current_dir / output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    summary_data = []

    print("\nAnalyzing datasets...")

    for name, df in df_dict.items():
        mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

        facility_id_col = _find_facility_id_column(df.columns)
        measure_id_col = _find_measure_id_column(df.columns)
        value_col = _find_likely_value_column(df.columns)

        unique_facilities = None
        if facility_id_col is not None:
            try:
                unique_facilities = df[facility_id_col].nunique(dropna=True)
            except Exception:
                unique_facilities = None

        potential_hospital_file = (
            facility_id_col is not None
            and measure_id_col is not None
            and value_col is not None
        )

        summary_data.append({
            "Dataset_Name": name,
            "Row_Count": df.shape[0],
            "Column_Count": df.shape[1],
            "Has_Facility_ID": facility_id_col is not None,
            "Facility_ID_Column": facility_id_col,
            "Unique_Facility_IDs": unique_facilities,
            "Has_Measure_ID": measure_id_col is not None,
            "Measure_ID_Column": measure_id_col,
            "Likely_Value_Column": value_col,
            "Potential_Hospital_File": potential_hospital_file,
            "Memory_MB": round(mem_mb, 2),
            "Columns_List": ", ".join(list(df.columns)),
        })

    summary_df = pd.DataFrame(summary_data)

    # Sort to put the most likely hospital modeling candidates near the top
    summary_df = summary_df.sort_values(
        by=["Potential_Hospital_File", "Has_Facility_ID", "Row_Count"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    report_file = out_path / "dataset_inventory_profile.csv"
    summary_df.to_csv(report_file, index=False)

    print(f"\n[SUCCESS] Data interpretation profile saved to: {report_file}")
    print("[INFO] Review this CSV to identify facility-level hospital files,")
    print("       likely feature sources, and low-value summary datasets.")

    return summary_df


if __name__ == "__main__":
    from importlib import import_module

    step_01 = import_module("01_data_import")
    all_data = step_01.load_all_raw_data()

    if all_data:
        profile_df = generate_data_profile(all_data)
        print(profile_df.head(10).to_string(index=False))