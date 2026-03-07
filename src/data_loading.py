import pandas as pd
from config import HAI_FILE, HCAHPS_FILE, TIMELY_FILE, GENERAL_FILE, HAC_FILE
from pathlib import Path
from config import INTERIM_DIR

hai = pd.read_csv(HAI_FILE, dtype={"Facility ID": "string"})
hcahps = pd.read_csv(HCAHPS_FILE, dtype={"Facility ID": "string"}, low_memory=False)
timely = pd.read_csv(TIMELY_FILE, dtype={"Facility ID": "string"}, low_memory=False)
general = pd.read_csv(GENERAL_FILE, dtype={"Facility ID": "string"})
hac = pd.read_csv(HAC_FILE, dtype={"Facility ID": "string"})

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

print("\nHAC columns:")
for col in hac.columns:
    print(col)

print("\nPayment Reduction value counts:")
print(hac["Payment Reduction"].value_counts(dropna=False))

hac_target = hac[["Facility ID", "Payment Reduction"]].copy()
hac_target = hac_target.dropna(subset=["Payment Reduction"])
hac_target["target"] = hac_target["Payment Reduction"].map({"No": 0, "Yes": 1})

print("\nTarget shape:", hac_target.shape)
print("\nTarget value counts:")
print(hac_target["target"].value_counts(dropna=False))


print("\nTarget unique Facility IDs:", hac_target["Facility ID"].nunique())
print("Duplicate Facility IDs in target:", hac_target["Facility ID"].duplicated().sum())


model_base = general.merge(hac_target, on="Facility ID", how="inner")

print("\nModel base shape:", model_base.shape)
print("Model base unique Facility IDs:", model_base["Facility ID"].nunique())


INTERIM_DIR.mkdir(parents=True, exist_ok=True)

output_path = INTERIM_DIR / "model_base.csv"
model_base.to_csv(output_path, index=False)

print("\nSaved model base to:")
print(output_path)


print("\nModel base columns:")
for col in model_base.columns:
    print(col)


print("\nHCAHPS rows per Facility ID summary:")
print(hcahps["Facility ID"].value_counts().describe())


print("\nHCAHPS columns:")
for col in hcahps.columns:
    print(col)


print("\nUnique HCAHPS Measure IDs:", hcahps["HCAHPS Measure ID"].nunique())
print("\nTop HCAHPS Measure ID counts:")
print(hcahps["HCAHPS Measure ID"].value_counts().head(20))


print("\nNon-null counts in HCAHPS value columns:")
print("Patient Survey Star Rating:", hcahps["Patient Survey Star Rating"].notna().sum())
print("HCAHPS Answer Percent:", hcahps["HCAHPS Answer Percent"].notna().sum())
print("HCAHPS Linear Mean Value:", hcahps["HCAHPS Linear Mean Value"].notna().sum())


print("\nSample values from HCAHPS value columns:")
print("\nPatient Survey Star Rating:")
print(hcahps["Patient Survey Star Rating"].astype(str).value_counts().head(10))

print("\nHCAHPS Answer Percent:")
print(hcahps["HCAHPS Answer Percent"].astype(str).value_counts().head(10))

print("\nHCAHPS Linear Mean Value:")
print(hcahps["HCAHPS Linear Mean Value"].astype(str).value_counts().head(10))


hcahps["star_rating_num"] = pd.to_numeric(hcahps["Patient Survey Star Rating"], errors="coerce")
hcahps["answer_percent_num"] = pd.to_numeric(hcahps["HCAHPS Answer Percent"], errors="coerce")
hcahps["linear_mean_num"] = pd.to_numeric(hcahps["HCAHPS Linear Mean Value"], errors="coerce")

print("\nNumeric HCAHPS value counts after coercion:")
print("star_rating_num:", hcahps["star_rating_num"].notna().sum())
print("answer_percent_num:", hcahps["answer_percent_num"].notna().sum())
print("linear_mean_num:", hcahps["linear_mean_num"].notna().sum())


hcahps_measure_summary = (
    hcahps.groupby("HCAHPS Measure ID")[["star_rating_num", "answer_percent_num", "linear_mean_num"]]
    .apply(lambda x: x.notna().sum())
)

print("\nHCAHPS measure summary:")
print(hcahps_measure_summary.head(20))


hcahps["hcahps_value"] = (
    hcahps["answer_percent_num"]
    .combine_first(hcahps["linear_mean_num"])
    .combine_first(hcahps["star_rating_num"])
)

print("\nUnified HCAHPS numeric values:", hcahps["hcahps_value"].notna().sum())
print(
    "Duplicate Facility ID + HCAHPS Measure ID rows:",
    hcahps.duplicated(subset=["Facility ID", "HCAHPS Measure ID"]).sum()
)


hcahps_wide = hcahps.pivot(
    index="Facility ID",
    columns="HCAHPS Measure ID",
    values="hcahps_value"
).reset_index()

print("\nHCAHPS wide shape:", hcahps_wide.shape)
print("HCAHPS wide unique Facility IDs:", hcahps_wide["Facility ID"].nunique())


model_with_hcahps = model_base.merge(hcahps_wide, on="Facility ID", how="left")

print("\nModel with HCAHPS shape:", model_with_hcahps.shape)
print("Model with HCAHPS unique Facility IDs:", model_with_hcahps["Facility ID"].nunique())


hcahps_feature_cols = [col for col in hcahps_wide.columns if col != "Facility ID"]
model_with_hcahps["hcahps_nonnull_count"] = model_with_hcahps[hcahps_feature_cols].notna().sum(axis=1)

print("\nHCAHPS non-null feature count summary:")
print(model_with_hcahps["hcahps_nonnull_count"].describe())


print("\nHCAHPS completeness counts:")
print(model_with_hcahps["hcahps_nonnull_count"].value_counts().sort_index())

print("\nHospitals with zero HCAHPS features:")
print((model_with_hcahps["hcahps_nonnull_count"] == 0).sum())


print("\nTarget by HCAHPS completeness:")
print(
    pd.crosstab(
        model_with_hcahps["hcahps_nonnull_count"],
        model_with_hcahps["target"],
        margins=True
    )
)


print("\nTimely rows per Facility ID summary:")
print(timely["Facility ID"].value_counts().describe())


print("\nTimely columns:")
for col in timely.columns:
    print(col)


print("\nUnique Timely Measure IDs:", timely["Measure ID"].nunique())
print("\nTop Timely Measure ID counts:")
print(timely["Measure ID"].value_counts().head(20))


print("\nTop Timely Score values:")
print(timely["Score"].astype(str).value_counts().head(20))


timely["score_num"] = pd.to_numeric(timely["Score"], errors="coerce")

print("\nNumeric Timely score count:", timely["score_num"].notna().sum())
print("Total Timely rows:", len(timely))


timely_measure_summary = timely.groupby("Measure ID")["score_num"].apply(lambda x: x.notna().sum())

print("\nTimely numeric score counts by Measure ID:")
print(timely_measure_summary.sort_values(ascending=False))


timely_numeric = timely[timely["score_num"].notna()].copy()

timely_wide = timely_numeric.pivot(
    index="Facility ID",
    columns="Measure ID",
    values="score_num"
).reset_index()

print("\nTimely wide shape:", timely_wide.shape)
print("Timely wide unique Facility IDs:", timely_wide["Facility ID"].nunique())


model_with_hcahps_timely = model_with_hcahps.merge(timely_wide, on="Facility ID", how="left")

print("\nModel with HCAHPS + Timely shape:", model_with_hcahps_timely.shape)
print("Model with HCAHPS + Timely unique Facility IDs:", model_with_hcahps_timely["Facility ID"].nunique())


timely_feature_cols = [col for col in timely_wide.columns if col != "Facility ID"]
model_with_hcahps_timely["timely_nonnull_count"] = model_with_hcahps_timely[timely_feature_cols].notna().sum(axis=1)

print("\nTimely non-null feature count summary:")
print(model_with_hcahps_timely["timely_nonnull_count"].describe())


print("\nTimely completeness counts:")
print(model_with_hcahps_timely["timely_nonnull_count"].value_counts().sort_index())

print("\nHospitals with zero Timely features:")
print((model_with_hcahps_timely["timely_nonnull_count"] == 0).sum())


print("\nTarget by Timely completeness:")
print(
    pd.crosstab(
        model_with_hcahps_timely["timely_nonnull_count"],
        model_with_hcahps_timely["target"],
        margins=True
    )
)


output_path_2 = INTERIM_DIR / "model_with_hcahps_timely.csv"
model_with_hcahps_timely.to_csv(output_path_2, index=False)

print("\nSaved model with HCAHPS + Timely to:")
print(output_path_2)


print("\nFinal merged file shape:", model_with_hcahps_timely.shape)
print("Final merged file columns:", len(model_with_hcahps_timely.columns))