import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def _ensure_dir(path_like) -> Path:
    path = Path(path_like).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_dataframe(df: pd.DataFrame, output_path: Path, label: str):
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Saved {label} to: {output_path}")


def run_eda(
    input_dir="../data/processed",
    figures_dir="../reports/figures",
    tables_dir="../data/processed",
):
    """
    Run lightweight exploratory data analysis on the master modeling dataset.

    Outputs
    -------
    1. Class distribution plot
    2. At-risk rate by hospital type
    3. At-risk rate by hospital ownership
    4. Missingness summary table for structural variables

    Parameters
    ----------
    input_dir : str, default="../data/processed"
        Relative path containing master_model_data.csv
    figures_dir : str, default="../reports/figures"
        Relative path for saving figures
    tables_dir : str, default="../data/processed"
        Relative path for saving summary tables

    Returns
    -------
    dict
        Dictionary containing key EDA DataFrames.
    """
    current_dir = Path(__file__).parent
    master_file = (current_dir / input_dir / "master_model_data.csv").resolve()
    figures_path = _ensure_dir(current_dir / figures_dir)
    tables_path = _ensure_dir(current_dir / tables_dir)

    df = pd.read_csv(master_file)

    eda_outputs = {}

    # ------------------------------------------------------------
    # 1. Class distribution
    # ------------------------------------------------------------
    class_counts = (
        df["HAI_at_risk"]
        .value_counts(dropna=False)
        .rename_axis("HAI_at_risk")
        .reset_index(name="count")
        .sort_values("HAI_at_risk")
        .reset_index(drop=True)
    )
    class_counts["percent"] = class_counts["count"] / class_counts["count"].sum()

    eda_outputs["class_counts"] = class_counts
    _save_dataframe(
        class_counts,
        tables_path / "eda_class_distribution.csv",
        "class distribution table"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(class_counts["HAI_at_risk"].astype(str), class_counts["count"])
    ax.set_title("HAI Risk Distribution")
    ax.set_xlabel("HAI_at_risk")
    ax.set_ylabel("Facility Count")
    fig.tight_layout()
    fig.savefig(figures_path / "eda_hai_risk_distribution.png", dpi=300)
    plt.show()
    print(f"[SUCCESS] Saved figure to: {figures_path / 'eda_hai_risk_distribution.png'}")

    # ------------------------------------------------------------
    # 2. At-risk rate by hospital type
    # ------------------------------------------------------------
    type_summary = (
        df.groupby("Hospital Type", dropna=False)
        .agg(
            facility_count=("Facility ID", "count"),
            at_risk_count=("HAI_at_risk", "sum"),
            at_risk_rate=("HAI_at_risk", "mean"),
        )
        .reset_index()
        .sort_values(["at_risk_rate", "facility_count"], ascending=[False, False])
        .reset_index(drop=True)
    )

    eda_outputs["type_summary"] = type_summary
    _save_dataframe(
        type_summary,
        tables_path / "eda_hospital_type_summary.csv",
        "hospital type summary table"
    )

    plot_type = type_summary.sort_values("at_risk_rate", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_type["Hospital Type"].astype(str), plot_type["at_risk_rate"])
    ax.set_title("At-Risk Rate by Hospital Type")
    ax.set_xlabel("At-Risk Rate")
    ax.set_ylabel("Hospital Type")
    fig.tight_layout()
    fig.savefig(figures_path / "eda_at_risk_rate_by_hospital_type.png", dpi=300)
    plt.show()
    print(f"[SUCCESS] Saved figure to: {figures_path / 'eda_at_risk_rate_by_hospital_type.png'}")

    # ------------------------------------------------------------
    # 3. At-risk rate by hospital ownership
    # ------------------------------------------------------------
    ownership_summary = (
        df.groupby("Hospital Ownership", dropna=False)
        .agg(
            facility_count=("Facility ID", "count"),
            at_risk_count=("HAI_at_risk", "sum"),
            at_risk_rate=("HAI_at_risk", "mean"),
        )
        .reset_index()
        .sort_values(["at_risk_rate", "facility_count"], ascending=[False, False])
        .reset_index(drop=True)
    )

    eda_outputs["ownership_summary"] = ownership_summary
    _save_dataframe(
        ownership_summary,
        tables_path / "eda_hospital_ownership_summary.csv",
        "hospital ownership summary table"
    )

    plot_owner = ownership_summary.sort_values("at_risk_rate", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(plot_owner["Hospital Ownership"].astype(str), plot_owner["at_risk_rate"])
    ax.set_title("At-Risk Rate by Hospital Ownership")
    ax.set_xlabel("At-Risk Rate")
    ax.set_ylabel("Hospital Ownership")
    fig.tight_layout()
    fig.savefig(figures_path / "eda_at_risk_rate_by_hospital_ownership.png", dpi=300)
    plt.show()
    print(f"[SUCCESS] Saved figure to: {figures_path / 'eda_at_risk_rate_by_hospital_ownership.png'}")

    # ------------------------------------------------------------
    # 4. Missingness summary for structural fields
    # ------------------------------------------------------------
    structural_cols = [
        "Facility ID",
        "HAI_at_risk",
        "Hospital Type",
        "Hospital Ownership",
        "Hospital overall rating",
    ]

    missingness_summary = pd.DataFrame({
        "column": structural_cols,
        "missing_count": [df[col].isna().sum() for col in structural_cols],
        "missing_percent": [df[col].isna().mean() for col in structural_cols],
    })

    eda_outputs["missingness_summary"] = missingness_summary
    _save_dataframe(
        missingness_summary,
        tables_path / "eda_structural_missingness_summary.csv",
        "structural missingness summary table"
    )

    print("[SUCCESS] Step 04 EDA complete.")
    return eda_outputs


if __name__ == "__main__":
    outputs = run_eda()
    for name, out_df in outputs.items():
        print(f"\n{name}")
        print(out_df.head().to_string(index=False))