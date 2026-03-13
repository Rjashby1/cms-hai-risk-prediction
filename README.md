# CMS HAI Risk Prediction

## Project Overview
This project leverages CMS hospital quality datasets to develop a machine learning pipeline that predicts whether a facility is at elevated risk for Healthcare-Associated Infection (HAI) safety outcomes. By integrating disparate clinical and administrative datasets, we aim to provide a data-driven early warning for hospital safety administrators.

## Objective
Build a predictive model using hospital characteristics, patient experience (HCAHPS) measures, and care-process indicators to identify facilities at higher risk of adverse HAI-related outcomes.

## Research Question
Can publicly available, high-dimensional CMS hospital quality data be used to predict facility-level HAI risk through automated feature mining?

## Data Source
CMS Provider Data Catalog hospital downloadable databases (Current Release: Jan 2026).
- Healthcare Associated Infections (HAI_1 through HAI_6)
- HCAHPS (Patient Surveys)
- Timely and Effective Care (Sepsis, Immunization)
- Hospital General Information
- HAC Reduction Program files

## Modeling Target
**Primary Target (`HAI_at_risk`):** A binary indicator (1/0) where 1 represents a facility performing "Worse than the National Benchmark" on **any** of the six primary HAI Standardized Infection Ratio (SIR) measures.

## Repository Structure
```text
cms-hai-risk-prediction/
├── README.md               # Project documentation
├── .gitignore              # Configured to ignore data/ and code/archive/
├── environment.yml         # Conda env with XGBoost, LightGBM, Optuna
├── main_pipeline.ipynb     # Central orchestrator for the project
├── code/                   # Modular Python scripts
│   ├── 00_download_data.py # Automated data retrieval from CMS
│   ├── 01_data_import.py   # Bulk ingestion of 70+ CSVs
│   ├── 02_data_interpretation.py
│   ├── 03_data_processing.py # Ground truth construction
│   ├── 04_data_analysis.py   # Automated EDA & visualization
│   └── archive/            # Local-only research scripts (ignored)
├── data/                   # (Ignored by Git)
│   ├── raw/                # Original CMS CSVs
│   ├── interim/            # Inventory profiles
│   └── processed/          # Final modeling datasets
└── models/                 # Saved .pkl and .joblib models (ignored)