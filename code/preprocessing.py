import pandas as pd
from pathlib import Path
import gdown

#Community area data 

script_dir = Path(__file__).parent
community_area_path = script_dir / "../data/raw-data/community_area.csv"
df_ca = pd.read_csv(community_area_path)
print(df_ca.head())

# Download the 311 request data from Google Drive
file_id = "1rYJpNKT4kix_NAPhL--LJIDq9Ctp1vhs"
url = f"https://drive.google.com/uc?id={file_id}"

gdown.download(url, "311_request.csv", quiet=False)

df_311 = pd.read_csv("311_request.csv")

print(df_311.head())

#Clean the database


df_ca.columns = df_ca.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
df_311.columns = df_311.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

if "status" in df_311.columns:
    df_311 = df_311[
        df_311["status"].astype(str).str.lower().isin(["completed", "closed"])
    ].copy()

if "created_date" in df_311.columns:
    created_col = "created_date"
elif "creation_date" in df_311.columns:
    created_col = "creation_date"
else:
    raise ValueError("No encontré columna de fecha de creación en 311_request.csv")

if "completion_date" in df_311.columns:
    done_col = "completion_date"
elif "closed_date" in df_311.columns:
    done_col = "closed_date"
else:
    raise ValueError("No encontré columna de fecha de cierre/completion en 311_request.csv")

if "community_area" not in df_311.columns:
    raise ValueError("No encontré columna 'community_area' en 311_request.csv")

df_311["created_at"] = pd.to_datetime(df_311[created_col], errors="coerce")
df_311["done_at"] = pd.to_datetime(df_311[done_col], errors="coerce")

df_311["response_time_hours"] = (
    (df_311["done_at"] - df_311["created_at"]).dt.total_seconds() / 3600
)

df_311["community_area"] = pd.to_numeric(df_311["community_area"], errors="coerce")


df_311 = df_311.dropna(subset=["community_area", "created_at", "done_at", "response_time_hours"])

df_311 = df_311[df_311["response_time_hours"] >= 0]

agg_311 = (
    df_311.groupby("community_area", as_index=False)
    .agg(
        total_requests=("response_time_hours", "size"),
        avg_response_time=("response_time_hours", "mean"),
    )
)

agg_311["total_requests"] = agg_311["total_requests"].fillna(0)

print(agg_311.head())
print("filas agregadas:", len(agg_311))

derived_dir = script_dir / "../data/derived-data"
derived_dir.mkdir(parents=True, exist_ok=True)

output_path = derived_dir / "community_area_analysis.csv"
agg_311.to_csv(output_path, index=False)
print(f"Save in: {output_path}")

if "community_area_name" in df_311.columns and "community_area" in df_ca.columns and "total_population" in df_ca.columns:
    df_ca_merge = df_ca[["community_area", "total_population"]].copy()
    df_ca_merge["community_area_name"] = df_ca_merge["community_area"].astype(str).str.strip().str.upper()

    df_ca_merge["total_population"] = pd.to_numeric(
        df_ca_merge["total_population"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )

    agg_name = (
        df_311.assign(
            community_area_name=df_311["community_area_name"].astype(str).str.strip().str.upper()
        )
        .groupby("community_area_name", as_index=False)
        .agg(
            total_requests=("response_time_hours", "size"),
            avg_response_time=("response_time_hours", "mean"),
        )
    )

    df_final = df_ca_merge.merge(agg_name, on="community_area_name", how="left")

    df_final["total_requests"] = df_final["total_requests"].fillna(0)
    df_final.loc[df_final["total_requests"] == 0, "avg_response_time"] = pd.NA

    df_final["requests_per_1000"] = (
        df_final["total_requests"] / df_final["total_population"]
    ) * 1000
    df_final.loc[
        df_final["total_population"].isna() | (df_final["total_population"] == 0),
        "requests_per_1000",
    ] = pd.NA

    output_pop = derived_dir / "community_area_analysis_with_population.csv"
    df_final.to_csv(output_pop, index=False)
    print(f"Guardado en: {output_pop}")
    print(df_final.head())
else:
    print(
        "No name or population columns found in the datasets, skipping merge with population data."
    )
