# Data Directory

This project uses publicly available hospital datasets from the CMS Provider Data Catalog.

## Source
CMS Provider Data Catalog — Hospitals  
https://data.cms.gov/provider-data/topics/hospitals

## Expected Raw Files
Download the complete ZIP archive for the current period (e.g., `hospitals_current_data.zip`) and extract it entirely into `data/raw/[release_date]/`. 
Do not delete any CSVs manually. The project pipeline will programmatically evaluate all available CSVs to identify potential predictive features for ALL HAI measures before discarding irrelevant data.

## Folder Structure
- `raw/` — original extracted CMS files (unaltered)
- `interim/` — cleaned, merged intermediate files, and data profile reports
- `processed/` — final modeling-ready dataset split into train/test CSVs

## Notes
- Raw data must remain strictly read-only.
- Download dates and CMS refresh periods are documented for reproducibility.