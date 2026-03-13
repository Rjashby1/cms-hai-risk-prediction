import pandas as pd
from pathlib import Path
import os

def load_all_raw_data(data_dir="../data/raw/cms_hospitals_jan2026"):
    """
    Dynamically loads ALL CSV files found in the target directory.
    Returns a dictionary of pandas DataFrames.
    """
    current_dir = Path(__file__).parent
    data_path = (current_dir / data_dir).resolve()
    
    data_frames = {}
    
    print(f"Scanning directory: {data_path}\n" + "-"*50)
    
    if not data_path.exists():
        print(f"[ERROR] Path does not exist: {data_path}")
        return data_frames

    # Find all CSV files in the directory
    csv_files = list(data_path.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files. Beginning import...\n")
    
    for filepath in csv_files:
        name = filepath.stem # Gets the filename without the .csv extension
        try:
            # Load with low_memory=False to handle mixed types in large files
            df = pd.read_csv(filepath, low_memory=False)
            data_frames[name] = df
            print(f"[LOADED] {name[:40]:<40} | Rows: {df.shape[0]:<7} | Cols: {df.shape[1]}")
        except Exception as e:
            print(f"[FAILED] {name} - Error: {e}")
            
    return data_frames

if __name__ == "__main__":
    df_dict = load_all_raw_data()