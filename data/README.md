# Data Directory

This project uses publicly available hospital datasets from the CMS Provider Data Catalog.

## Source
CMS Provider Data Catalog — Hospitals  
https://data.cms.gov/provider-data/topics/hospitals

## Expected Raw Files
Place the following files in `data/raw/`:

- `Healthcare_Associated_Infections-Hospital.csv`
- `HCAHPS-Hospital.csv`
- `Timely_and_Effective_Care-Hospital.csv`
- `Hospital_General_Information.csv`
- `FY_2026_HAC_Reduction_Program_Hospital.csv`  
  or the most current HAC Reduction Program file available
- `HOSPITAL_Data_Dictionary.pdf`

## Folder Structure
- `raw/` — original downloaded CMS files
- `interim/` — cleaned and merged intermediate files
- `processed/` — final modeling-ready datasets

## Notes
- Raw data should not be modified directly.
- Download dates and CMS refresh periods should be documented for reproducibility.
- Archived CMS hospital releases may be used for longitudinal or time-aware modeling.