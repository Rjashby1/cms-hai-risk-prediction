import pandas as pd
from config import HAI_FILE, HCAHPS_FILE, TIMELY_FILE, GENERAL_FILE, HAC_FILE

hai = pd.read_csv(HAI_FILE)
hcahps = pd.read_csv(HCAHPS_FILE, low_memory=False)
timely = pd.read_csv(TIMELY_FILE, low_memory=False)
general = pd.read_csv(GENERAL_FILE)
hac = pd.read_csv(HAC_FILE)

print("HAI shape:", hai.shape)
print("HCAHPS shape:", hcahps.shape)
print("Timely shape:", timely.shape)
print("General shape:", general.shape)
print("HAC shape:", hac.shape)

print("HAI has Facility ID:", "Facility ID" in hai.columns)
print("HCAHPS has Facility ID:", "Facility ID" in hcahps.columns)
print("Timely has Facility ID:", "Facility ID" in timely.columns)
print("General has Facility ID:", "Facility ID" in general.columns)
print("HAC has Facility ID:", "Facility ID" in hac.columns)

print("HAI unique Facility IDs:", hai["Facility ID"].nunique())
print("HCAHPS unique Facility IDs:", hcahps["Facility ID"].nunique())
print("Timely unique Facility IDs:", timely["Facility ID"].nunique())
print("General unique Facility IDs:", general["Facility ID"].nunique())
print("HAC unique Facility IDs:", hac["Facility ID"].nunique())