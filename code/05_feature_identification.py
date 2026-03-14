import pandas as pd
import numpy as np
from pathlib import Path

def mine_features(df_dict, overlap_threshold=0.50, completion_threshold=0.50):
    """
    Automated feature discovery: Scans all datasets and selects predictors 
    based on population overlap and data density.
    """
    current_dir = Path(__file__).parent
    master_path = (current_dir / "../data/processed/master_model_data.csv").resolve()
    master_df = pd.read_csv(master_path)
    master_df['Facility ID'] = master_df['Facility ID'].astype(str).str.zfill(6)
    
    target_ids = set(master_df['Facility ID'])
    print(f"Targeting predictors for {len(target_ids)} facilities...")

    # Exclude known non-data files to save CPU cycles
    skip_keywords = ['Footnote', 'State', 'National', 'ZIP', 'Date', 'Crosswalk']

    for name, df in df_dict.items():
        if any(key in name for key in skip_keywords): continue
        if 'Facility ID' not in df.columns: continue
        
        # 1. Overlap Check
        df['Facility ID'] = df['Facility ID'].astype(str).str.zfill(6)
        overlap_count = len(set(df['Facility ID']) & target_ids)
        if (overlap_count / len(target_ids)) < overlap_threshold:
            continue # Skip files that don't cover enough of our hospitals (50% threshold)

        try:
            # 2. Find the 'Data' column
            val_col = next((c for c in ['Score', 'Value', 'Numeric Score', 'HCAHPS Linear Mean Score'] if c in df.columns), None)
            if not val_col: continue

            # 3. Pivot and Numeric Check
            temp_df = df.copy()
            temp_df[val_col] = pd.to_numeric(temp_df[val_col], errors='coerce')
            
            pivot_df = temp_df.pivot_table(
                index='Facility ID',
                columns='Measure ID',
                values=val_col,
                aggfunc='mean'
            ).reset_index()

            # 4. Completion Threshold
            # Drop columns that are mostly NaNs for our specific 1,432 hospitals (50% threshold)
            limit = len(pivot_df) * completion_threshold
            pivot_df = pivot_df.dropna(axis=1, thresh=limit)

            if pivot_df.shape[1] > 1:
                master_df = pd.merge(master_df, pivot_df, on='Facility ID', how='left')
                print(f"[DATA DISCOVERED] {name}: Added {pivot_df.shape[1]-1} features.")

        except Exception as e:
            print(f"[FAIL] {name}: {e}")

    # Fill remaining gaps with median to prevent model crashes
    numeric_cols = master_df.select_dtypes(include=[np.number]).columns
    master_df[numeric_cols] = master_df[numeric_cols].fillna(master_df[numeric_cols].median())

    # Save final modeling dataset
    final_path = (current_dir / "../data/processed/final_modeling_feature_set.csv").resolve()
    master_df.to_csv(final_path, index=False)
    
    print("-" * 30)
    print(f"[SUCCESS] Final Feature Set Shape: {master_df.shape}")
    return master_df