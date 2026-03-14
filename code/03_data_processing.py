import pandas as pd
import numpy as np
from pathlib import Path

def build_target_and_master(df_dict, output_dir="../data/processed"):
    """
    Constructs the ground truth labels based on Arithmetic Risk (SIR > 1.0).
    """
    # 1. Isolate the HAI data
    hai_df = df_dict["Healthcare_Associated_Infections-Hospital"].copy()

    # 2. Narrow down to SIR measures (HAI_1_SIR, etc.)
    # Handles strings that may have extra spaces or different formats by using 'contains' instead of strict equality.
    mask = hai_df['Measure ID'].str.contains('HAI', na=False) & \
           hai_df['Measure ID'].str.contains('SIR', na=False)
    target_prep = hai_df[mask].copy()

    # 3. Numeric Conversion
    # Coerces non-numeric values to NaN, which will be treated as not at risk (0) in the next step.
    target_prep['Score_Numeric'] = pd.to_numeric(target_prep['Score'], errors='coerce')
    target_prep['is_worse'] = (target_prep['Score_Numeric'] > 1.0).astype(int)

    # 4. Aggregate to Facility Level: Flag 1 if ANY SIR > 1.0
    gt_df = target_prep.groupby('Facility ID')['is_worse'].max().reset_index()
    gt_df.rename(columns={'is_worse': 'HAI_at_risk'}, inplace=True)

    # 5. ID Alignment
    # Forces IDs to match even if loaded as different types (Str vs Int)
    gt_df['Facility ID'] = gt_df['Facility ID'].astype(str).str.zfill(6)
    
    gen_df = df_dict["Hospital_General_Information"].copy()
    gen_df['Facility ID'] = gen_df['Facility ID'].astype(str).str.zfill(6)

    # 6. Merge with Structural Data
    cols = ['Facility ID', 'Hospital Type', 'Hospital Ownership', 'Hospital overall rating']
    master_df = pd.merge(gt_df, gen_df[cols], on='Facility ID', how='inner')

    # 7. Save and Report
    current_dir = Path(__file__).parent
    out_path = (current_dir / output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(out_path / "master_model_data.csv", index=False)

    print(f"--- 03 Processing Complete ---")
    print(f"Total Facilities: {len(master_df)}")
    print(f"At Risk Count:   {master_df['HAI_at_risk'].sum()}")
    
    return master_df