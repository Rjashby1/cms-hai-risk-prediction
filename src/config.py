from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CMS_RAW_DIR = RAW_DIR / "cms_hospitals_jan2026"

HAI_FILE = CMS_RAW_DIR / "Healthcare_Associated_Infections-Hospital.csv"
HCAHPS_FILE = CMS_RAW_DIR / "HCAHPS-Hospital.csv"
TIMELY_FILE = CMS_RAW_DIR / "Timely_and_Effective_Care-Hospital.csv"
GENERAL_FILE = CMS_RAW_DIR / "Hospital_General_Information.csv"
HAC_FILE = CMS_RAW_DIR / "FY_2026_HAC_Reduction_Program_Hospital.csv"
DATA_DICTIONARY_FILE = CMS_RAW_DIR / "HOSPITAL_Data_Dictionary.pdf"

INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"