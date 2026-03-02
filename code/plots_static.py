from pathlib import Path

import altair as alt
import pandas as pd


SCRIPT_DIR = Path(__file__).parent
DERIVED_DIR = SCRIPT_DIR / "../data/derived-data"
PLOTS_DIR = DERIVED_DIR / "plots"
TYPE_PATH = DERIVED_DIR / "df_311_type.csv"
AREA_PATH = DERIVED_DIR / "df_311_ca.csv"

EXCLUDED_AREAS = {"OHARE"}
EXCLUDED_SERVICE_TYPES = {
    "311 INFORMATION ONLY CALL",
    "Tree Trim Request (NO LONGER BEING ACCEPTED)",
}
QUARTILE_LABELS = [
    "Low Income",
    "Middle-Low Income",
    "Middle-High Income",
    "High Income",
]
CHICAGO_BAR_COLORS = [
    "#41B6E6",
    "#C8102E",
    "#0B1F41",
    "#9BD3F5",
    "#7A0019",
    "#6FA8DC",
]
CHICAGO_HEAT_COLORS = [
    "#F7FBFF",
    "#D6EAF8",
    "#9ECAE1",
    "#4A90C2",
    "#0B3C6D",
]


def ensure_input(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")



def add_income_groups(df: pd.DataFrame) -> pd.DataFrame:
    quartile_source = (
        df[["community_area", "income_estimate"]]
        .drop_duplicates(subset=["community_area"])
        .copy()
    )
    quartile_source["income_group"] = pd.qcut(
        quartile_source["income_estimate"],
        4,
        labels=QUARTILE_LABELS,
    )

    return df.merge(
        quartile_source[["community_area", "income_group"]],
        on="community_area",
        how="left",
    )



def load_type_data() -> pd.DataFrame:
    ensure_input(TYPE_PATH)
    df = pd.read_csv(TYPE_PATH)

    for col in ["community_area", "total_requests", "income_estimate", "total_population"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "community_area",
            "community_area_name",
            "service_request_type",
            "income_estimate",
            "total_population",
            "total_requests",
        ]
    ).copy()
    df = df[df["total_population"] > 0]
    df = df[~df["community_area_name"].isin(EXCLUDED_AREAS)]
    df = df[~df["service_request_type"].isin(EXCLUDED_SERVICE_TYPES)]
    return add_income_groups(df)



def load_area_data() -> pd.DataFrame:
    ensure_input(AREA_PATH)
    df = pd.read_csv(AREA_PATH)

    for col in ["community_area", "income_estimate", "total_population", "requests_per_1000"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "community_area",
            "community_area_name",
            "income_estimate",
            "total_population",
            "requests_per_1000",
        ]
    ).copy()
    df = df[df["total_population"] > 0]
    df = df[~df["community_area_name"].isin(EXCLUDED_AREAS)]
    df = add_income_groups(df)
    df["income_group"] = pd.Categorical(
        df["income_group"], categories=QUARTILE_LABELS, ordered=True
    )
    return df



def summarize_service_by_quartile(df: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    top_services = (
        df.groupby("service_request_type")["total_requests"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    summary = (
        df[df["service_request_type"].isin(top_services)]
        .groupby(["income_group", "service_request_type"], observed=False)
        .agg(
            total_requests=("total_requests", "sum"),
            total_population=("total_population", "sum"),
        )
        .reset_index()
    )
    summary["requests_per_1000"] = (
        summary["total_requests"] / summary["total_population"]
    ) * 1000
    summary["income_group"] = pd.Categorical(
        summary["income_group"], categories=QUARTILE_LABELS, ordered=True
    )
    return summary.sort_values(["income_group", "requests_per_1000"], ascending=[True, False])



def build_heatmap(summary: pd.DataFrame) -> alt.Chart:
    service_order = (
        summary.groupby("service_request_type")["requests_per_1000"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    return (
        alt.Chart(summary)
        .mark_rect()
        .encode(
            x=alt.X("income_group:N", sort=QUARTILE_LABELS, title="Income Group"),
            y=alt.Y("service_request_type:N", sort=service_order, title="Service Type"),
            color=alt.Color(
                "requests_per_1000:Q",
                title="Requests per 1,000 residents",
                scale=alt.Scale(range=CHICAGO_HEAT_COLORS),
            ),
            tooltip=[
                alt.Tooltip("income_group:N", title="Income Group"),
                alt.Tooltip("service_request_type:N", title="Service Type"),
                alt.Tooltip("requests_per_1000:Q", title="Requests per 1,000", format=".2f"),
                alt.Tooltip("total_requests:Q", title="Total Requests", format=","),
            ],
        )
        .properties(
            width=420,
            height=260,
            title="Heatmap of 311 Service Types by Income Group",
        )
        .configure_axis(labelFontSize=12, titleFontSize=14)
        .configure_legend(titleFontSize=12, labelFontSize=11)
        .configure_title(fontSize=16)
    )


def build_boxplot(area_df: pd.DataFrame) -> alt.Chart:
    plot_df = area_df[area_df["community_area_name"] != "NEAR WEST SIDE"].copy()

    return (
        alt.Chart(plot_df)
        .mark_boxplot(extent=1.5, size=36)
        .encode(
            x=alt.X("income_group:N", sort=QUARTILE_LABELS, title="Income Group"),
            y=alt.Y("requests_per_1000:Q", title="Requests per 1,000 residents"),
            color=alt.Color(
                "income_group:N",
                sort=QUARTILE_LABELS,
                scale=alt.Scale(
                    domain=QUARTILE_LABELS,
                    range=["#9BD3F5", "#41B6E6", "#C8102E", "#0B1F41"],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("community_area_name:N", title="Community Area"),
                alt.Tooltip("income_group:N", title="Income Group"),
                alt.Tooltip("requests_per_1000:Q", title="Requests per 1,000", format=".2f"),
            ],
        )
        .properties(
            width=420,
            height=360,
            title="Distribution of 311 Request Rates by Income Group",
        )
        .configure_axis(labelFontSize=11, titleFontSize=14)
        .configure_axisX(labelFontSize=9, labelAngle=0)
        .configure_title(fontSize=16)
    )



def save_chart(chart: alt.Chart, stem: str) -> None:
    html_path = PLOTS_DIR / f"{stem}.html"
    png_path = PLOTS_DIR / f"{stem}.png"
    chart.save(html_path)
    chart.save(png_path, scale_factor=3)
    print(f"Saved plot: {html_path}")
    print(f"Saved plot: {png_path}")



def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    type_df = load_type_data()
    area_df = load_area_data()
    summary = summarize_service_by_quartile(type_df)

    save_chart(build_heatmap(summary), "heatmap_income_services")
    save_chart(build_boxplot(area_df), "box_requests_income")


if __name__ == "__main__":
    main()
