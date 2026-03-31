import pandas as pd
from pathlib import Path


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of the DataFrame with stripped column names.

    This keeps the original CMS column wording intact while removing
    accidental leading or trailing whitespace that could cause join
    or lookup issues later in the pipeline.
    """
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def load_all_raw_data(data_dir="../data/raw/cms_hospitals_jan2026"):
    """
    Dynamically load all CSV files found in the target raw-data directory.

    Parameters
    ----------
    data_dir : str, default="../data/raw/cms_hospitals_jan2026"
        Relative path to the folder containing the extracted CMS CSV files.

    Returns
    -------
    dict
        Dictionary of pandas DataFrames keyed by file stem.
    """
    current_dir = Path(__file__).parent
    data_path = (current_dir / data_dir).resolve()

    data_frames = {}

    print(f"Scanning directory: {data_path}")
    print("-" * 50)

    if not data_path.exists():
        print(f"[ERROR] Path does not exist: {data_path}")
        return data_frames

    csv_files = sorted(data_path.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files. Beginning import...\n")

    for filepath in csv_files:
        dataset_name = filepath.stem

        try:
            df = pd.read_csv(filepath, low_memory=False)
            df = _clean_column_names(df)

            data_frames[dataset_name] = df

            print(
                f"[LOADED] {dataset_name[:40]:<40} | "
                f"Rows: {df.shape[0]:<7} | Cols: {df.shape[1]}"
            )

        except Exception as e:
            print(f"[FAILED] {dataset_name} - Error: {e}")

    print("\n[SUCCESS] Raw data import complete.")
    print(f"[INFO] Loaded {len(data_frames)} datasets into memory.")

    return data_frames


if __name__ == "__main__":
    df_dict = load_all_raw_data()
    print(f"Datasets loaded: {len(df_dict)}")