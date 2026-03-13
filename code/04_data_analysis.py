import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def run_eda(input_dir="../data/processed"):
    current_dir = Path(__file__).parent
    master_file = (current_dir / input_dir / "master_model_data.csv").resolve()
    
    df = pd.read_csv(master_file)
    
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='HAI_at_risk', hue='HAI_at_risk', palette='viridis', legend=False)
    plt.title('HAI Risk Distribution (0=Safe, 1=At Risk)')
    plt.show() # This ensures it pops up in your .ipynb cell