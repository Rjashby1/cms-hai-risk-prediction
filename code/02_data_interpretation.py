import pandas as pd
from pathlib import Path
import importlib
di = importlib.import_module("01_data_import")

def generate_data_profile(df_dict, output_dir="../data/interim"):
    """
    Analyzes all loaded DataFrames and exports a summary report to the interim folder.
    """
    current_dir = Path(__file__).parent
    out_path = (current_dir / output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True) # Create interim folder if it doesn't exist
    
    summary_data = []
    
    print("\nAnalyzing datasets...")
    for name, df in df_dict.items():
        # Get memory usage in MB
        mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        # Check if it has a Facility ID column (crucial for joining data later)
        has_facility_id = any(col.lower() in ['facility id', 'provider id'] for col in df.columns)
        
        summary_data.append({
            "Dataset_Name": name,
            "Row_Count": df.shape[0],
            "Column_Count": df.shape[1],
            "Has_Facility_ID": has_facility_id,
            "Memory_MB": round(mem_mb, 2),
            "Columns_List": ", ".join(list(df.columns))
        })
        
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by Row Count to easily spot national vs facility-level datasets
    summary_df = summary_df.sort_values(by="Row_Count", ascending=False)
    
    # Save the profile report
    report_file = out_path / "dataset_inventory_profile.csv"
    summary_df.to_csv(report_file, index=False)
    
    print(f"\n[SUCCESS] Data interpretation profile saved to: {report_file}")
    print("Review this CSV to determine which files to cull in Step 03.")
    
    return summary_df

if __name__ == "__main__":
    # 1. Import all data using the module we just built
    all_data = di.load_all_raw_data()
    
    # 2. Run the interpretation profile
    if all_data:
        profile_df = generate_data_profile(all_data)