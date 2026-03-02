from pathlib import Path

import altair as alt
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pydeck as pdk
import numpy as np
import pandas as pd
import streamlit as st


SCRIPT_DIR = Path(__file__).parent
TYPE_DATA_PATH = SCRIPT_DIR / "../data/derived-data/df_311_type.csv"
GEO_PATH = SCRIPT_DIR / "../data/derived-data/Boundaries_-_Community_Areas_20260301.geojson"
MIN_MEDIAN_DAYS = 0.25


def fit_slope(group: pd.DataFrame) -> float:
    if len(group) < 3:
        return float("nan")
    x = group["income_estimate"].to_numpy()
    y = group["avg_response_days"].to_numpy()
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


@st.cache_data
def load_data() -> pd.DataFrame:
    if not TYPE_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data file: {TYPE_DATA_PATH}")

    df = pd.read_csv(TYPE_DATA_PATH)

    required_cols = [
        "community_area",
        "community_area_name",
        "service_request_type",
        "income_estimate",
        "avg_response_time",
        "total_requests",
        "requests_per_1000_by_type",
        "total_population",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in df_311_type.csv: {missing}")

    for col in [
        "community_area",
        "income_estimate",
        "avg_response_time",
        "total_requests",
        "requests_per_1000_by_type",
        "total_population",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["community_area", "community_area_name", "service_request_type", "income_estimate", "avg_response_time"]
    ).copy()
    df = df[(df["income_estimate"] > 0) & (df["avg_response_time"] >= 0)].copy()

    # Exclude non-comparable information-only request type.
    df = df[~df["service_request_type"].astype(str).str.upper().eq("311 INFORMATION ONLY CALL")].copy()

    # Exclude OHARE from analysis.
    df = df[~df["community_area_name"].astype(str).str.upper().eq("OHARE")].copy()

    df["community_area"] = df["community_area"].astype(int)
    df["avg_response_days"] = df["avg_response_time"] / 24.0

    labels = ["Low income", "Lower-middle income", "Upper-middle income", "High income"]
    df["income_group"] = pd.qcut(df["income_estimate"], q=4, labels=labels)
    return df


@st.cache_data
def load_boundaries() -> gpd.GeoDataFrame:
    if not GEO_PATH.exists():
        raise FileNotFoundError(f"Missing boundary file: {GEO_PATH}")

    gdf = gpd.read_file(GEO_PATH)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)

    if "area_num_1" in gdf.columns:
        gdf["community_area"] = pd.to_numeric(gdf["area_num_1"], errors="coerce")
    elif "area_numbe" in gdf.columns:
        gdf["community_area"] = pd.to_numeric(gdf["area_numbe"], errors="coerce")
    else:
        raise ValueError("Boundary file missing area number field (area_num_1 or area_numbe).")

    gdf = gdf.dropna(subset=["community_area"]).copy()
    gdf["community_area"] = gdf["community_area"].astype(int)

    return gdf


def build_scatter(filtered: pd.DataFrame) -> alt.Chart:
    color_scale = alt.Scale(
        domain=["Low income", "Lower-middle income", "Upper-middle income", "High income"],
        range=["#4C92C3", "#4CAF50", "#F28E2B", "#E15759"],
    )

    y_min = float(filtered["avg_response_days"].min())
    y_max = float(filtered["avg_response_days"].max())
    y_pad = max((y_max - y_min) * 0.10, 0.25)
    y_domain = [max(0.0, y_min - y_pad), y_max + y_pad]

    points = (
        alt.Chart(filtered)
        .mark_circle(size=90, opacity=0.8, stroke="white", strokeWidth=0.7)
        .encode(
            x=alt.X("income_estimate:Q", title="Estimated Household Income (USD)"),
            y=alt.Y(
                "avg_response_days:Q",
                title="Average Response Time (days)",
                scale=alt.Scale(domain=y_domain),
                axis=alt.Axis(format=".2f"),
            ),
            color=alt.Color("income_group:N", title="Income Group", scale=color_scale),
            tooltip=[
                alt.Tooltip("community_area_name:N", title="Community Area"),
                alt.Tooltip("service_request_type:N", title="Service Type"),
                alt.Tooltip("income_estimate:Q", title="Income", format=",.0f"),
                alt.Tooltip("avg_response_days:Q", title="Response Time (days)", format=",.2f"),
                alt.Tooltip("total_requests:Q", title="Requests", format=",.0f"),
            ],
        )
    )

    trend_line = (
        alt.Chart(filtered)
        .transform_regression("income_estimate", "avg_response_days", method="linear")
        .mark_line(color="#222222", size=2.5)
        .encode(x="income_estimate:Q", y="avg_response_days:Q")
    )

    return (points + trend_line).properties(height=540)


def build_map_figure(gdf_boundaries: gpd.GeoDataFrame, map_df: pd.DataFrame, metric_col: str, map_mode: str) -> pdk.Deck:
    merged = gdf_boundaries.merge(
        map_df[["community_area", "community_area_name", "top_service_request", "income_group_area", metric_col]],
        on="community_area",
        how="left",
    ).copy()

    # Build color scale for choropleth.
    values = merged[metric_col].astype(float)
    valid = values.dropna()
    if len(valid) == 0:
        merged["r"], merged["g"], merged["b"] = 220, 220, 220
    else:
        vmin, vmax = valid.min(), valid.max()
        if vmax == vmin:
            norm = np.zeros(len(merged))
        else:
            norm = ((values - vmin) / (vmax - vmin)).fillna(0.0).clip(0, 1)

        cmap_name = "Blues" if metric_col == "total_requests_overall" else "Oranges"
        cmap = cm.get_cmap(cmap_name)
        colors = norm.apply(lambda x: cmap(float(x)))
        merged["r"] = colors.apply(lambda c: int(c[0] * 255))
        merged["g"] = colors.apply(lambda c: int(c[1] * 255))
        merged["b"] = colors.apply(lambda c: int(c[2] * 255))

    metric_label = "Request volume" if metric_col == "total_requests_overall" else "Avg response time (days)"
    merged["metric_value"] = merged[metric_col].round(2)
    merged["metric_label"] = metric_label
    merged["top_service_request"] = merged["top_service_request"].fillna("N/A")
    merged["income_group_area"] = merged["income_group_area"].fillna("N/A")

    geojson_data = merged.__geo_interface__

    layer = pdk.Layer(
        "GeoJsonLayer",
        data=geojson_data,
        pickable=True,
        stroked=True,
        filled=True,
        extruded=False,
        wireframe=False,
        get_fill_color="[properties.r, properties.g, properties.b, 190]",
        get_line_color=[80, 80, 80, 200],
        line_width_min_pixels=1,
    )

    view_state = pdk.ViewState(latitude=41.8781, longitude=-87.6298, zoom=9.35, pitch=0)

    return pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="light",
        tooltip={
            "html": "<b>{community_area_name}</b><br/>Income group: {income_group_area}<br/>Top service: {top_service_request}<br/>{metric_label}: {metric_value}",
            "style": {"backgroundColor": "#1f2937", "color": "white", "fontSize": "12px"},
        },
    )


def main() -> None:
    st.set_page_config(page_title="Chicago 311 Dashboard", layout="wide")
    st.title("Mapping 311 Requests and Response Times in Chicago Dashboard (2023-2024)")

    df = load_data()
    gdf_boundaries = load_boundaries()

    # Scatter filters (only for scatter)
    slope_df = (
        df.groupby("service_request_type")
        .apply(
            lambda g: pd.Series(
                {
                    "slope": fit_slope(g),
                    "requests": g["total_requests"].sum(),
                    "median_days": g["avg_response_days"].median(),
                }
            )
        )
        .reset_index()
    )

    slope_df = slope_df.dropna(subset=["slope", "median_days"]).copy()
    slope_df = slope_df[slope_df["median_days"] >= MIN_MEDIAN_DAYS].copy()

    trend_option = st.radio("Trend filter (scatter only)", ["Negative trend", "Positive trend"], horizontal=True)

    if trend_option == "Negative trend":
        trend_df = slope_df[slope_df["slope"] < 0].copy()
    else:
        trend_df = slope_df[slope_df["slope"] > 0].copy()

    trend_df["trend_strength"] = trend_df["slope"].abs()
    trend_df = trend_df.sort_values(["trend_strength", "requests"], ascending=[False, False])
    scatter_services = trend_df["service_request_type"].tolist()

    if not scatter_services:
        st.warning("No service request types match this trend filter.")
        st.stop()

    selected_service = st.selectbox("Service request type (scatter only)", scatter_services)
    scatter_df = df[df["service_request_type"] == selected_service].copy()

    st.subheader("Income vs Response Time")
    st.altair_chart(build_scatter(scatter_df), use_container_width=True)

    st.subheader("Chicago Community Area Map")
    map_mode = st.radio(
        "Map metric",
        ["Service demand intensity (request volume)", "Service supply efficiency (response time)"],
        horizontal=True,
    )

    map_agg = (
        df.groupby(["community_area", "community_area_name"], as_index=False)
        .agg(total_requests_overall=("total_requests", "sum"))
    )

    # Top 1 service request per community area (by request volume).
    top_service = (
        df.groupby(["community_area", "service_request_type"], as_index=False)["total_requests"]
        .sum()
        .sort_values(["community_area", "total_requests"], ascending=[True, False])
        .drop_duplicates(subset=["community_area"])
        .rename(columns={"service_request_type": "top_service_request"})[["community_area", "top_service_request"]]
    )

    map_agg = map_agg.merge(top_service, on="community_area", how="left")

    # Income group classification per community area.
    area_income_group = (
        df.groupby("community_area", as_index=False)
        .agg(income_group_area=("income_group", "first"))
    )
    area_income_group["income_group_area"] = area_income_group["income_group_area"].astype(str)
    map_agg = map_agg.merge(area_income_group, on="community_area", how="left")

    # Response-time map excludes FOREST GLEN only for the efficiency metric.
    tmp = df[~df["community_area_name"].astype(str).str.upper().eq("FOREST GLEN")].copy()
    tmp["weighted_days"] = tmp["avg_response_days"] * tmp["total_requests"]
    weighted = (
        tmp.groupby("community_area", as_index=False)
        .agg(weighted_days_sum=("weighted_days", "sum"), total_requests=("total_requests", "sum"))
    )
    weighted["avg_response_days_overall"] = weighted["weighted_days_sum"] / weighted["total_requests"].where(weighted["total_requests"] > 0)

    map_agg = map_agg.merge(weighted[["community_area", "avg_response_days_overall"]], on="community_area", how="left")

    metric_col = "total_requests_overall" if map_mode == "Service demand intensity (request volume)" else "avg_response_days_overall"

    map_deck = build_map_figure(gdf_boundaries, map_agg, metric_col, map_mode)
    st.pydeck_chart(map_deck, use_container_width=True, height=430)

    st.caption("Excluded: OHARE and 311 INFORMATION ONLY CALL")
    st.caption("Response-time map additionally excludes: FOREST GLEN, is excluded only from the response-time map because it is dominated by a few unusually slow service categories, creating an outlier that compresses the color scale for other community areas.")
    st.caption("Map reading: darker areas indicate higher values; lighter areas indicate lower values.")
    st.caption("Map source: Boundaries_-_Community_Areas_20260301.geojson")


if __name__ == "__main__":
    main()
