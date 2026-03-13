import pandas as pd
from pathlib import Path

def build_target_and_master(df_dict, output_dir="../data/processed"):
    current_dir = Path(__file__).parent
    out_path = (current_dir / output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Isolate HAI data
    hai_df = df_dict["Healthcare_Associated_Infections-Hospital"].copy()
    
    # Target: Flag 'Worse' comparisons for HAI_1 through HAI_6
    risk_string = "Worse than the National Benchmark"
    mask = hai_df['Measure ID'].str.contains('HAI_[1-6]_SIR', na=False)
    target_prep = hai_df[mask].copy()

    # Convert to binary
    target_prep['is_worse'] = (target_prep['Compared to National'] == risk_string).astype(int)

    # Group by Facility: 1 if Worse in ANY category, 0 otherwise
    gt_df = target_prep.groupby('Facility ID')['is_worse'].max().reset_index()
    gt_df.rename(columns={'is_worse': 'HAI_at_risk'}, inplace=True)

    # Base merge with General Info
    gen_df = df_dict["Hospital_General_Information"][['Facility ID', 'Hospital Type', 'Hospital Ownership', 'Hospital overall rating']]
    master_df = pd.merge(gt_df, gen_df, on='Facility ID', how='inner')
    
    master_df.to_csv(out_path / "master_model_data.csv", index=False)
    print(f"[SUCCESS] Master Data Created. At Risk Count: {master_df['HAI_at_risk'].sum()}")
    return master_df