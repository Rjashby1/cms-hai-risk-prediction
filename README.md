# CMS HAI Risk Prediction

## Project Overview
This project develops an end-to-end machine learning pipeline using CMS hospital quality datasets to identify facilities at elevated risk for Healthcare-Associated Infection (HAI) outcomes. The goal is to move beyond static reporting and provide a data-driven screening tool that supports facility-level self-assessment and targeted quality improvement.

## Objective
Build a reproducible pipeline that ingests raw CMS hospital data, constructs a facility-level HAI risk target, mines predictive features across multiple datasets, and trains models capable of identifying high-risk facilities and the factors contributing to that risk.

## Research Question
Can publicly available, high-dimensional CMS hospital quality data be used to reliably predict facility-level HAI risk and provide actionable insight for hospital leadership?

## Data Source
CMS Provider Data Catalog hospital downloadable databases (Current Release: Jan 2026).

Key datasets used:
- Healthcare Associated Infections (HAI SIR measures)
- Timely and Effective Care
- Complications and Deaths
- Unplanned Hospital Visits
- Outpatient Imaging Efficiency
- Medicare Spending per Beneficiary
- Hospital General Information

## Modeling Target
**Primary Target (`HAI_at_risk`)**  
A binary indicator (1/0) where a facility is labeled "at risk" if it performs worse than the national benchmark (SIR > 1.0) on any reported HAI measure. This frames HAI risk as a facility-level vulnerability rather than a single outcome.

## Pipeline Overview

The project is structured as a modular pipeline executed through `main_pipeline.ipynb`:

- **00–01:** Automated data retrieval and dynamic ingestion of CMS datasets  
- **02:** Dataset profiling and feature source identification  
- **03:** Target construction and master modeling dataset creation  
- **04:** Exploratory analysis and cohort validation  
- **05:** Dynamic feature mining across eligible hospital datasets  
- **05b:** Feature governance and pruning (manual exclusions + correlation filtering)  
- **06–08:** Model training and tuning  
  - Logistic Regression (interpretable baseline)  
  - Random Forest (primary model)  
  - XGBoost (boosted ensemble comparison)  
- **09:** Held-out evaluation and interpretation  
  - ROC-AUC, precision-recall, balanced accuracy  
  - Confusion matrices and threshold analysis  
  - Feature importance and permutation importance  
- **10:** Model application to all CMS facilities  
  - Generates a Tableau-ready dataset with predicted HAI risk

## Model Selection
Random Forest achieved the strongest performance on the held-out test set and was selected as the final model for deployment.

## Final Output
The pipeline produces a facility-level dataset:

`all_cms_facility_hai_risk_predictions.csv`

This includes:
- Facility identifiers and metadata  
- Predicted HAI risk (Yes/No)  
- Predicted probability of risk  
- Risk tier classification (Low / Medium / High)  
- Prediction status (including facilities with insufficient data)

This output is designed for downstream use in dashboards (e.g., Tableau) and supports a facility lookup and risk-screening use case.

## Dashboard
Accompanying the facility-level dataset is a Tableau dashboard which explores specific facility performance and various features associated with HAIs. This dashboard is accessible as a Tableau workbook within this repository and as a URL hosted on Tableau Public at https://public.tableau.com/app/profile/alysa.pugmire/viz/HealthcareAssociatedInfection/Dashboard 

## Repository Structure
```text
cms-hai-risk-prediction/
├── README.md
├── .gitignore
├── environment.yml
├── main_pipeline.ipynb
├── code/
│   ├── 00_download_data.py
│   ├── 01_data_import.py
│   ├── 02_data_interpretation.py
│   ├── 03_data_processing.py
│   ├── 04_data_analysis.py
│   ├── 05_feature_identification.py
│   ├── 05b_feature_pruning.py
│   ├── 06_model_logreg.py
│   ├── 07_model_randforest.py
│   ├── 08_model_xgboost.py
│   ├── 09_evaluator.py
│   ├── 10_model_application.py
│   └── archive/
├── dashboard/
|   └── Healthcare_Associated_Infection.txbx
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/
|   └── index.html
├── reports/
│   └── figures/
└── models/
