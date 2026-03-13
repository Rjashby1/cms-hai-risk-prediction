import pandas as pd
import numpy as np
from pathlib import Path

def mine_features(df_dict, min_completion_pct=0.50):
    """
    Scans all datasets for numeric measures, pivots 'Long' data to 'Wide',
    and merges everything into a high-dimensional feature set.
    """
    # Start with our Ground Truth from Step 03
    current_dir = Path(__file__).parent
    processed_path = (current_dir / "../data/processed/master_model_data.csv").resolve()
    
    if not processed_path.exists():
        print("[ERROR] master_model_data.csv not found. Run Step 03 first.")
        return None
    
    master_df = pd.read_csv(processed_path)
    initial_count = len(master_df)

    # List of files we want to skip (metadata, national averages, etc.)
    skip_keywords = ['National', 'State', 'Footnote', 'Date', 'Crosswalk', 'Geocoded']

    print(f"Beginning Feature Mining across {len(df_dict)} datasets...")

    for name, df in df_dict.items():
        # 1. Skip non-hospital-level files
        if any(key in name for key in skip_keywords):
            continue
            
        # 2. Ensure 'Facility ID' exists for joining
        if 'Facility ID' not in df.columns:
            continue

        # 3. Identify potential value columns
        # CMS usually uses 'Score', 'Measure Value', or 'Patient Survey Star Rating'
        val_cols = [c for c in df.columns if any(k in c for k in ['Score', 'Value', 'Rating', 'Percent'])]
        
        if not val_cols or 'Measure ID' not in df.columns:
            continue

        # 4. Pivot 'Long' to 'Wide'
        # This turns Measure IDs (rows) into Columns
        try:
            # We take the first available value column
            val_col = val_cols[0]
            
            # Clean numeric values (strip strings like "Not Available")
            df_clean = df.copy()
            df_clean[val_col] = pd.to_numeric(df_clean[val_col], errors='coerce')
            
            # Pivot
            pivot_df = df_clean.pivot_table(
                index='Facility ID',
                columns='Measure ID',
                values=val_col,
                aggfunc='first'
            ).reset_index()

            # 5. Filter Sparsity: Drop columns that are mostly empty
            limit = len(pivot_df) * min_completion_pct
            pivot_df = pivot_df.dropna(axis=1, thresh=limit)

            # 6. Merge into Master
            if pivot_df.shape[1] > 1: # Ensure we actually have feature columns
                master_df = pd.merge(master_df, pivot_df, on='Facility ID', how='left')
                print(f"[MINED] {name}: Added {pivot_df.shape[1]-1} features.")

        except Exception as e:
            print(f"[SKIP] {name} failed to pivot: {e}")

    # Final Cleaning: Drop columns with ZERO variance (all same value)
    master_df = master_df.loc[:, master_df.nunique() > 1]

    # Save final modeling dataset
    final_path = (current_dir / "../data/processed/final_modeling_feature_set.csv").resolve()
    master_df.to_csv(final_path, index=False)
    
    print("-" * 50)
    print(f"[SUCCESS] Feature Mining Complete.")
    print(f"Final Shape: {master_df.shape} (Rows, Columns)")
    print(f"File Saved: {final_path}")
    
    return master_df

if __name__ == "__main__":
    # This allows standalone testing if needed
    import importlib
    di = importlib.import_module("01_data_import")
    raw = di.load_all_raw_data()
    mine_features(raw)