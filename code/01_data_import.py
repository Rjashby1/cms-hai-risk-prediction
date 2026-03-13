import pandas as pd
from pathlib import Path

def load_raw_data(data_dir="../data"):
    """
    Loads the four primary CMS CSV files from the specified data directory.
    Returns a dictionary of pandas DataFrames.
    """
    # Resolve the path relative to where the script is being executed
    data_path = Path(data_dir).resolve()
    
    files = {
        "hai": data_path / "Healthcare_Associated_Infections-Hospital.csv",
        "hcahps": data_path / "HCAHPS-Hospital.csv",
        "timely": data_path / "Timely_and_Effective_Care-Hospital.csv",
        "general": data_path / "Hospital_General_Information.csv",
    }
    
    data_frames = {}
    
    print(f"Looking for data in: {data_path}\n" + "-"*40)
    
    for name, filepath in files.items():
        try:
            # low_memory=False prevents mixed-type warnings on large CMS datasets
            data_frames[name] = pd.read_csv(filepath, low_memory=False)
            print(f"[SUCCESS] Loaded '{name}': {data_frames[name].shape[0]} rows, {data_frames[name].shape[1]} columns")
        except FileNotFoundError:
            print(f"[ERROR] Could not find {filepath.name}")
            
    return data_frames

if __name__ == "__main__":
    # This block only runs if you execute this specific file directly in the terminal
    print("Testing data import module...\n")
    df_dict = load_raw_data()