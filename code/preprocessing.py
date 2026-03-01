import re
from pathlib import Path

import gdown
import pandas as pd

script_dir = Path(__file__).parent


def normalize_name(value: str) -> str:
    text = str(value).strip().upper()
    text = re.sub(r"[^A-Z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_col(name: str) -> str:
    value = name.strip().lower()
    value = value.replace("$", "")
    value = value.replace("+", "plus")
    value = value.replace(" to ", "_")
    value = value.replace(",", "")
    value = value.replace(" ", "_")
    value = value.replace("-", "_")
    value = re.sub(r"[^a-z0-9_]", "", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


community_area_lookup = {
    1: "ROGERS PARK", 2: "WEST RIDGE", 3: "UPTOWN", 4: "LINCOLN SQUARE", 5: "NORTH CENTER",
    6: "LAKE VIEW", 7: "LINCOLN PARK", 8: "NEAR NORTH SIDE", 9: "EDISON PARK", 10: "NORWOOD PARK",
    11: "JEFFERSON PARK", 12: "FOREST GLEN", 13: "NORTH PARK", 14: "ALBANY PARK", 15: "PORTAGE PARK",
    16: "IRVING PARK", 17: "DUNNING", 18: "MONTCLARE", 19: "BELMONT CRAGIN", 20: "HERMOSA",
    21: "AVONDALE", 22: "LOGAN SQUARE", 23: "HUMBOLDT PARK", 24: "WEST TOWN", 25: "AUSTIN",
    26: "WEST GARFIELD PARK", 27: "EAST GARFIELD PARK", 28: "NEAR WEST SIDE", 29: "NORTH LAWNDALE", 30: "SOUTH LAWNDALE",
    31: "LOWER WEST SIDE", 32: "LOOP", 33: "NEAR SOUTH SIDE", 34: "ARMOUR SQUARE", 35: "DOUGLAS",
    36: "OAKLAND", 37: "FULLER PARK", 38: "GRAND BOULEVARD", 39: "KENWOOD", 40: "WASHINGTON PARK",
    41: "HYDE PARK", 42: "WOODLAWN", 43: "SOUTH SHORE", 44: "CHATHAM", 45: "AVALON PARK",
    46: "SOUTH CHICAGO", 47: "BURNSIDE", 48: "CALUMET HEIGHTS", 49: "ROSELAND", 50: "PULLMAN",
    51: "SOUTH DEERING", 52: "EAST SIDE", 53: "WEST PULLMAN", 54: "RIVERDALE", 55: "HEGEWISCH",
    56: "GARFIELD RIDGE", 57: "ARCHER HEIGHTS", 58: "BRIGHTON PARK", 59: "MCKINLEY PARK", 60: "BRIDGEPORT",
    61: "NEW CITY", 62: "WEST ELSDON", 63: "GAGE PARK", 64: "CLEARING", 65: "WEST LAWN",
    66: "CHICAGO LAWN", 67: "WEST ENGLEWOOD", 68: "ENGLEWOOD", 69: "GREATER GRAND CROSSING", 70: "ASHBURN",
    71: "AUBURN GRESHAM", 72: "BEVERLY", 73: "WASHINGTON HEIGHTS", 74: "MOUNT GREENWOOD", 75: "MORGAN PARK",
    76: "OHARE", 77: "EDGEWATER",
}
name_to_id = {name: area_id for area_id, name in community_area_lookup.items()}


# Community area dataset
community_area_path = script_dir / "../data/raw-data/community_areas.csv"
if not community_area_path.exists():
    raise FileNotFoundError(f"Missing input file: {community_area_path}")

raw_ca = pd.read_csv(community_area_path)
print(f"Community source: {community_area_path}")

# 311 dataset
requests_path = script_dir / "../data/raw-data/311_request.csv"
if not requests_path.exists():
    file_id = "1rYJpNKT4kix_NAPhL--LJIDq9Ctp1vhs"
    url = f"https://drive.google.com/uc?id={file_id}"
    requests_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(url, str(requests_path), quiet=False)

raw_311 = pd.read_csv(requests_path)
print(f"311 source: {requests_path}")

# Clean ACS/community area data
acs = raw_ca.copy()
acs.columns = [normalize_col(c) for c in acs.columns]

if "acs_year" in acs.columns:
    acs["acs_year"] = to_numeric(acs["acs_year"])
    latest_year = acs["acs_year"].max()
    if pd.notna(latest_year):
        acs = acs[acs["acs_year"] == latest_year].copy()

numeric_cols = [
    "under_25000",
    "25000_49999",
    "50000_74999",
    "75000_125000",
    "125000_plus",
    "male_0_to_17",
    "male_18_to_24",
    "male_25_to_34",
    "male_35_to_49",
    "male_50_to_64",
    "male_65",
    "female_0_to_17",
    "female_18_to_24",
    "female_25_to_34",
    "female_35_to_49",
    "female_50_to_64",
    "female_65_plus",
    "total_population",
    "white",
    "black_or_african_american",
    "american_indian_or_alaska_native",
    "asian",
    "native_hawaiian_or_pacific_islander",
    "other_race",
    "multiracial",
    "white_not_hispanic_or_latino",
    "hispanic_or_latino",
]
for col in numeric_cols:
    if col in acs.columns:
        acs[col] = to_numeric(acs[col])

if "community_area" not in acs.columns:
    raise ValueError("Expected column 'community_area' not found after normalization")

income_cols = ["under_25000", "25000_49999", "50000_74999", "75000_125000", "125000_plus"]
if all(c in acs.columns for c in income_cols):
    acs["total_households_est"] = acs[income_cols].sum(axis=1)
    midpoints = [12500, 37500, 62500, 100000, 150000]
    weighted_income = acs[income_cols].fillna(0).mul(midpoints, axis=1).sum(axis=1)
    acs["income_estimate"] = weighted_income / acs["total_households_est"].where(acs["total_households_est"] > 0)
else:
    acs["income_estimate"] = pd.NA

if "total_population" in acs.columns:
    for src, dst in [
        ("black_or_african_american", "pct_black"),
        ("hispanic_or_latino", "pct_hispanic"),
        ("white", "pct_white"),
        ("asian", "pct_asian"),
    ]:
        if src in acs.columns:
            acs[dst] = (acs[src] / acs["total_population"].where(acs["total_population"] > 0)) * 100

acs["community_area_name"] = acs["community_area"].map(normalize_name)
acs["community_area_number"] = acs["community_area_name"].map(name_to_id)

acs_keep_cols = [
    "acs_year",
    "community_area",
    "community_area_name",
    "community_area_number",
    "total_population",
    "total_households_est",
    "income_estimate",
    "pct_black",
    "pct_hispanic",
    "pct_white",
    "pct_asian",
]
acs_available_cols = [c for c in acs_keep_cols if c in acs.columns]
acs_filtered = acs[acs_available_cols].drop_duplicates(subset=["community_area_name"]).copy()
acs_filtered = acs_filtered.sort_values(
    by=[c for c in ["community_area_number", "community_area_name"] if c in acs_filtered.columns]
)

# Clean 311 data
raw_311.columns = raw_311.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

if "status" in raw_311.columns:
    raw_311 = raw_311[raw_311["status"].astype(str).str.lower().isin(["completed", "closed"])].copy()

if "created_date" in raw_311.columns:
    created_col = "created_date"
elif "creation_date" in raw_311.columns:
    created_col = "creation_date"
else:
    raise ValueError("No encontré columna de fecha de creación en 311")

if "completion_date" in raw_311.columns:
    done_col = "completion_date"
elif "closed_date" in raw_311.columns:
    done_col = "closed_date"
else:
    raise ValueError("No encontré columna de fecha de cierre/completion en 311")

if "community_area" not in raw_311.columns:
    raise ValueError("No encontré columna 'community_area' en 311")

if "sr_type" in raw_311.columns:
    service_col = "sr_type"
elif "service_request_type" in raw_311.columns:
    service_col = "service_request_type"
else:
    raise ValueError("No encontré columna de tipo de servicio en 311")

fmt = "%m/%d/%Y %I:%M:%S %p"
raw_311["created_at"] = pd.to_datetime(raw_311[created_col], format=fmt, errors="coerce")
raw_311["done_at"] = pd.to_datetime(raw_311[done_col], format=fmt, errors="coerce")

mask_created = raw_311["created_at"].isna() & raw_311[created_col].notna()
if mask_created.any():
    raw_311.loc[mask_created, "created_at"] = pd.to_datetime(
        raw_311.loc[mask_created, created_col], errors="coerce"
    )

mask_done = raw_311["done_at"].isna() & raw_311[done_col].notna()
if mask_done.any():
    raw_311.loc[mask_done, "done_at"] = pd.to_datetime(
        raw_311.loc[mask_done, done_col], errors="coerce"
    )

raw_311["response_time_hours"] = (raw_311["done_at"] - raw_311["created_at"]).dt.total_seconds() / 3600
raw_311["community_area"] = pd.to_numeric(raw_311["community_area"], errors="coerce")
raw_311["service_request_type"] = raw_311[service_col].astype(str).str.strip()

raw_311 = raw_311.dropna(
    subset=["community_area", "created_at", "done_at", "response_time_hours", "service_request_type"]
)
raw_311 = raw_311[raw_311["response_time_hours"] >= 0]
raw_311 = raw_311[raw_311["service_request_type"] != ""]

agg_311 = (
    raw_311.groupby("community_area", as_index=False)
    .agg(
        total_requests=("response_time_hours", "size"),
        avg_response_time=("response_time_hours", "mean"),
    )
)
agg_311["community_area"] = agg_311["community_area"].astype("Int64")
agg_311["community_area_name"] = agg_311["community_area"].map(community_area_lookup)

agg_311_type = (
    raw_311.groupby(["community_area", "service_request_type"], as_index=False)
    .agg(
        total_requests=("response_time_hours", "size"),
        avg_response_time=("response_time_hours", "mean"),
    )
)
agg_311_type["community_area"] = agg_311_type["community_area"].astype("Int64")
agg_311_type["community_area_name"] = agg_311_type["community_area"].map(community_area_lookup)

# Merge 311 with ACS summary
acs_merge = acs_filtered[[c for c in [
    "community_area_name",
    "community_area_number",
    "total_population",
    "income_estimate",
    "pct_black",
    "pct_hispanic",
    "pct_white",
    "pct_asian",
] if c in acs_filtered.columns]].copy()
acs_merge = acs_merge.rename(columns={"community_area_number": "community_area"})

merged = agg_311.merge(acs_merge, on=["community_area", "community_area_name"], how="left")
merged_type = agg_311_type.merge(acs_merge, on=["community_area", "community_area_name"], how="left")

if "total_population" in merged.columns:
    merged["requests_per_1000"] = (merged["total_requests"] / merged["total_population"]) * 1000
    merged.loc[
        merged["total_population"].isna() | (merged["total_population"] <= 0),
        "requests_per_1000",
    ] = pd.NA

if "total_population" in merged_type.columns:
    merged_type["requests_per_1000_by_type"] = (
        merged_type["total_requests"] / merged_type["total_population"]
    ) * 1000
    merged_type.loc[
        merged_type["total_population"].isna() | (merged_type["total_population"] <= 0),
        "requests_per_1000_by_type",
    ] = pd.NA

# Output
derived_dir = script_dir / "../data/derived-data"
derived_dir.mkdir(parents=True, exist_ok=True)

acs_output_path = derived_dir / "acs_filtered.csv"
df311_output_path = derived_dir / "df_311_ca.csv"
df311_type_output_path = derived_dir / "df_311_type.csv"

acs_filtered.to_csv(acs_output_path, index=False)
merged.to_csv(df311_output_path, index=False)
merged_type.to_csv(df311_type_output_path, index=False)

print(f"Saved: {acs_output_path}")
print(f"ACS rows: {len(acs_filtered)}")
print(f"Saved: {df311_output_path}")
print(f"311 rows: {len(merged)}")
print(f"Saved: {df311_type_output_path}")
print(f"311 type rows: {len(merged_type)}")
print(merged.head())
