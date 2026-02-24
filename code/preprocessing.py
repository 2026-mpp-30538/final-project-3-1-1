import re
from pathlib import Path

import gdown
import pandas as pd

script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent


def normalize_name(value: str) -> str:
    text = str(value).strip().upper()
    text = re.sub(r"[^A-Z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


#Community area dataset 
community_area_path = project_dir / "data/raw-data/community_areas.csv"
if not community_area_path.exists():
    raise FileNotFoundError("Error")

df_ca = pd.read_csv(community_area_path)
print(f"Community source: {community_area_path}")

# 311 dataset 
requests_path = project_dir / "data/raw-data/311_request.csv"
if not requests_path.exists():
    file_id = "1rYJpNKT4kix_NAPhL--LJIDq9Ctp1vhs"
    url = f"https://drive.google.com/uc?id={file_id}"
    requests_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(url, str(requests_path), quiet=False)

df_311 = pd.read_csv(requests_path)
print(f"311 source: {requests_path}")

#Cleaning 
df_ca.columns = df_ca.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
df_311.columns = df_311.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

# Keep completed/closed requests
if "status" in df_311.columns:
    df_311 = df_311[df_311["status"].astype(str).str.lower().isin(["completed", "closed"])].copy()

# Date columns
if "created_date" in df_311.columns:
    created_col = "created_date"
elif "creation_date" in df_311.columns:
    created_col = "creation_date"
else:
    raise ValueError("No encontré columna de fecha de creación en 311")

if "completion_date" in df_311.columns:
    done_col = "completion_date"
elif "closed_date" in df_311.columns:
    done_col = "closed_date"
else:
    raise ValueError("No encontré columna de fecha de cierre/completion en 311")

if "community_area" not in df_311.columns:
    raise ValueError("No encontré columna 'community_area' en 311")

# Parse datetime
fmt = "%m/%d/%Y %I:%M:%S %p"
df_311["created_at"] = pd.to_datetime(df_311[created_col], format=fmt, errors="coerce")
df_311["done_at"] = pd.to_datetime(df_311[done_col], format=fmt, errors="coerce")

# Fallback parse for unexpected formats
mask_created = df_311["created_at"].isna() & df_311[created_col].notna()
if mask_created.any():
    df_311.loc[mask_created, "created_at"] = pd.to_datetime(
        df_311.loc[mask_created, created_col], errors="coerce"
    )

mask_done = df_311["done_at"].isna() & df_311[done_col].notna()
if mask_done.any():
    df_311.loc[mask_done, "done_at"] = pd.to_datetime(
        df_311.loc[mask_done, done_col], errors="coerce"
    )

df_311["response_time_hours"] = (df_311["done_at"] - df_311["created_at"]).dt.total_seconds() / 3600
df_311["community_area"] = pd.to_numeric(df_311["community_area"], errors="coerce")

df_311 = df_311.dropna(subset=["community_area", "created_at", "done_at", "response_time_hours"])
df_311 = df_311[df_311["response_time_hours"] >= 0]

# Aggregate 311
agg_311 = (
    df_311.groupby("community_area", as_index=False)
    .agg(
        total_requests=("response_time_hours", "size"),
        avg_response_time=("response_time_hours", "mean"),
    )
)
agg_311["community_area"] = agg_311["community_area"].astype("Int64")

#merge 
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

#AI discoulure, we use AI to set the name with the code of the community area. 


agg_311["community_area_name"] = agg_311["community_area"].map(community_area_lookup)
agg_311["community_key"] = agg_311["community_area_name"].map(normalize_name)

if "community_area" not in df_ca.columns:
    raise ValueError("No encontré columna 'community_area' en community areas")

# Keep latest ACS year if present
if "acs_year" in df_ca.columns:
    df_ca["acs_year"] = pd.to_numeric(df_ca["acs_year"], errors="coerce")
    latest_year = df_ca["acs_year"].max()
    if pd.notna(latest_year):
        df_ca = df_ca[df_ca["acs_year"] == latest_year].copy()

# Normalize key in ACS
df_ca["community_area_name"] = df_ca["community_area"].astype(str).map(normalize_name)
df_ca["community_key"] = df_ca["community_area_name"]

acs_num_cols = [
    "total_population",
    "under_$25,000",
    "$25,000_to_$49,999",
    "$50,000_to_$74,999",
    "$75,000_to_$125,000",
    "$125,000_+",
]
for col in acs_num_cols:
    if col in df_ca.columns:
        df_ca[col] = pd.to_numeric(df_ca[col].astype(str).str.replace(",", "", regex=False), errors="coerce")

# Income estimate from bins
income_bins = [
    "under_$25,000",
    "$25,000_to_$49,999",
    "$50,000_to_$74,999",
    "$75,000_to_$125,000",
    "$125,000_+",
]
if all(col in df_ca.columns for col in income_bins):
    midpoints = [12500, 37500, 62500, 100000, 150000]
    hh_counts = df_ca[income_bins].fillna(0)
    hh_total = hh_counts.sum(axis=1)
    df_ca["income_estimate"] = hh_counts.mul(midpoints, axis=1).sum(axis=1) / hh_total.where(hh_total > 0)
else:
    df_ca["income_estimate"] = pd.NA

keep_cols = ["community_key", "community_area_name", "income_estimate"]
if "total_population" in df_ca.columns:
    keep_cols.append("total_population")

df_ca_small = df_ca[keep_cols].drop_duplicates(subset=["community_key"]).copy()

# Merge final
merged = agg_311.merge(df_ca_small, on="community_key", how="left", suffixes=("", "_acs"))
merged["community_area_name"] = merged["community_area_name_acs"].fillna(merged["community_area_name"])
merged = merged.drop(columns=[c for c in ["community_area_name_acs", "community_key"] if c in merged.columns])

# Requests per 1000 
if "total_population" in merged.columns:
    merged["requests_per_1000"] = (merged["total_requests"] / merged["total_population"]) * 1000
    merged.loc[
        merged["total_population"].isna() | (merged["total_population"] <= 0),
        "requests_per_1000",
    ] = pd.NA


# Output

derived_dir = project_dir / "data/derived-data"
derived_dir.mkdir(parents=True, exist_ok=True)

output_path = derived_dir / "df_311_ca.csv"
merged.to_csv(output_path, index=False)

print(f"Saved: {output_path}")
print(f"Rows: {len(merged)}")
print(merged.head())
