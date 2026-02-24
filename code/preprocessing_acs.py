import re
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).parent
RAW_PATH = SCRIPT_DIR / "../data/raw-data/community_areas.csv"
OUTPUT_PATH = SCRIPT_DIR / "../data/derived-data/acs_filtered.csv"


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


def normalize_area_name(value: str) -> str:
    text = str(value).strip().upper()
    text = re.sub(r"[^A-Z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


if not RAW_PATH.exists():
    raise FileNotFoundError(f"Missing input file: {RAW_PATH}")


df = pd.read_csv(RAW_PATH)
df.columns = [normalize_col(c) for c in df.columns]

for col in ["acs_year"]:
    if col in df.columns:
        df[col] = to_numeric(df[col])

if "acs_year" in df.columns:
    latest_year = df["acs_year"].max()
    if pd.notna(latest_year):
        df = df[df["acs_year"] == latest_year].copy()

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
    if col in df.columns:
        df[col] = to_numeric(df[col])

if "community_area" not in df.columns:
    raise ValueError("Expected column 'community_area' not found after normalization")

income_cols = ["under_25000", "25000_49999", "50000_74999", "75000_125000", "125000_plus"]
if all(c in df.columns for c in income_cols):
    df["total_households_est"] = df[income_cols].sum(axis=1)
    midpoints = [12500, 37500, 62500, 100000, 150000]
    weighted_income = df[income_cols].fillna(0).mul(midpoints, axis=1).sum(axis=1)
    df["income_estimate"] = weighted_income / df["total_households_est"].where(df["total_households_est"] > 0)

if "total_population" in df.columns:
    for src, dst in [
        ("black_or_african_american", "pct_black"),
        ("hispanic_or_latino", "pct_hispanic"),
        ("white", "pct_white"),
        ("asian", "pct_asian"),
    ]:
        if src in df.columns:
            df[dst] = (df[src] / df["total_population"].where(df["total_population"] > 0)) * 100

# Build merge keys aligned with df_311_ca.csv
name_to_id = {
    "ROGERS PARK": 1, "WEST RIDGE": 2, "UPTOWN": 3, "LINCOLN SQUARE": 4, "NORTH CENTER": 5,
    "LAKE VIEW": 6, "LINCOLN PARK": 7, "NEAR NORTH SIDE": 8, "EDISON PARK": 9, "NORWOOD PARK": 10,
    "JEFFERSON PARK": 11, "FOREST GLEN": 12, "NORTH PARK": 13, "ALBANY PARK": 14, "PORTAGE PARK": 15,
    "IRVING PARK": 16, "DUNNING": 17, "MONTCLARE": 18, "BELMONT CRAGIN": 19, "HERMOSA": 20,
    "AVONDALE": 21, "LOGAN SQUARE": 22, "HUMBOLDT PARK": 23, "WEST TOWN": 24, "AUSTIN": 25,
    "WEST GARFIELD PARK": 26, "EAST GARFIELD PARK": 27, "NEAR WEST SIDE": 28, "NORTH LAWNDALE": 29,
    "SOUTH LAWNDALE": 30, "LOWER WEST SIDE": 31, "LOOP": 32, "NEAR SOUTH SIDE": 33, "ARMOUR SQUARE": 34,
    "DOUGLAS": 35, "OAKLAND": 36, "FULLER PARK": 37, "GRAND BOULEVARD": 38, "KENWOOD": 39,
    "WASHINGTON PARK": 40, "HYDE PARK": 41, "WOODLAWN": 42, "SOUTH SHORE": 43, "CHATHAM": 44,
    "AVALON PARK": 45, "SOUTH CHICAGO": 46, "BURNSIDE": 47, "CALUMET HEIGHTS": 48, "ROSELAND": 49,
    "PULLMAN": 50, "SOUTH DEERING": 51, "EAST SIDE": 52, "WEST PULLMAN": 53, "RIVERDALE": 54,
    "HEGEWISCH": 55, "GARFIELD RIDGE": 56, "ARCHER HEIGHTS": 57, "BRIGHTON PARK": 58, "MCKINLEY PARK": 59,
    "BRIDGEPORT": 60, "NEW CITY": 61, "WEST ELSDON": 62, "GAGE PARK": 63, "CLEARING": 64,
    "WEST LAWN": 65, "CHICAGO LAWN": 66, "WEST ENGLEWOOD": 67, "ENGLEWOOD": 68,
    "GREATER GRAND CROSSING": 69, "ASHBURN": 70, "AUBURN GRESHAM": 71, "BEVERLY": 72,
    "WASHINGTON HEIGHTS": 73, "MOUNT GREENWOOD": 74, "MORGAN PARK": 75, "OHARE": 76, "EDGEWATER": 77,
}

df["community_area_name"] = df["community_area"].map(normalize_area_name)
df["community_area_number"] = df["community_area_name"].map(name_to_id)

keep_cols = [
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

available_cols = [c for c in keep_cols if c in df.columns]
out = df[available_cols].drop_duplicates(subset=["community_area_name"]).copy()
out = out.sort_values(by=[c for c in ["community_area_number", "community_area_name"] if c in out.columns])

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUTPUT_PATH, index=False)

print(f"Saved: {OUTPUT_PATH}")
print(f"Rows: {len(out)}")
print(out.head())
