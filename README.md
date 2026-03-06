# CMS HAI Risk Prediction

## Project Overview
This project uses CMS hospital quality datasets to develop a machine learning model that predicts whether a hospital is at elevated risk for hospital-acquired infection (HAI)-related safety outcomes.

## Objective
Build a predictive model using hospital characteristics, patient experience measures, and care-process indicators to identify facilities at higher risk of adverse HAI-related outcomes.

## Research Question
Can publicly available CMS hospital quality data be used to predict future HAI-related risk at the facility level?

## Data Source
CMS Provider Data Catalog hospital downloadable databases:
- Healthcare Associated Infections
- HCAHPS
- Timely and Effective Care
- Hospital General Information
- HAC Reduction Program files
- Archived hospital releases for historical modeling

## Proposed Target
Primary target:
- Payment Reduction (binary) from HAC Reduction Program

Secondary targets:
- Total HAC Score
- Composite high-risk HAI label

## Methods
- Data cleaning and harmonization by Facility ID
- Feature engineering across hospital-level CMS tables
- Baseline models: logistic regression, random forest, gradient boosting
- Deep learning model: feedforward neural network for tabular classification
- Evaluation: ROC-AUC, PR-AUC, recall, precision, calibration

## Repository Structure
[brief folder explanation]

## Reproducibility
Raw data are not stored in this repository. See `data/README.md` for instructions on obtaining CMS source files.

## Team
- Robert Ashby
- [Add teammates]

## Status
In progress
