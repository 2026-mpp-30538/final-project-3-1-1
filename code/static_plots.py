from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "../data/derived-data/df_311_ca.csv"
PLOTS_DIR = SCRIPT_DIR / "../data/derived-data/plots"
OUT_PATH = PLOTS_DIR / "scatter_income_vs_avg_response_time_altair.html"


def iqr_bounds(series: pd.Series) -> tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing required file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = ["community_area_name", "income_estimate", "avg_response_time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input data: {missing}")

    df = df[required].copy()
    df["income_estimate"] = pd.to_numeric(df["income_estimate"], errors="coerce")
    df["avg_response_time"] = pd.to_numeric(df["avg_response_time"], errors="coerce")
    df = df.dropna(subset=["income_estimate", "avg_response_time"]).copy()

    income_lo, income_hi = iqr_bounds(df["income_estimate"])
    resp_lo, resp_hi = iqr_bounds(df["avg_response_time"])

    outlier_mask = (
        (df["income_estimate"] < income_lo)
        | (df["income_estimate"] > income_hi)
        | (df["avg_response_time"] < resp_lo)
        | (df["avg_response_time"] > resp_hi)
    )

    outliers = df.loc[outlier_mask, "community_area_name"].sort_values().tolist()

    print("Outlier community areas removed before plotting:")
    if outliers:
        for name in outliers:
            print(f"- {name}")
    else:
        print("- None")

    clean = df.loc[~outlier_mask].copy()

    labels = ["Low income", "Lower-middle income", "Upper-middle income", "High income"]
    clean["income_group"] = pd.qcut(clean["income_estimate"], q=4, labels=labels)

    # R^2 from linear regression: avg_response_time ~ income_estimate
    x = clean["income_estimate"].to_numpy()
    y = clean["avg_response_time"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    # Label 3 richest and 3 poorest (after outlier removal)
    poorest = clean.nsmallest(3, "income_estimate")
    richest = clean.nlargest(3, "income_estimate")
    labels_df = pd.concat([poorest, richest], ignore_index=True)

    color_scale = alt.Scale(
        domain=labels,
        range=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"],
    )

    base = alt.Chart(clean).encode(
        x=alt.X("income_estimate:Q", title="Estimated Household Income (USD)"),
        y=alt.Y("avg_response_time:Q", title="Average Response Time (hours)"),
        color=alt.Color("income_group:N", title="Income Groups", scale=color_scale),
        tooltip=[
            alt.Tooltip("community_area_name:N", title="Community Area"),
            alt.Tooltip("income_group:N", title="Income Group"),
            alt.Tooltip("income_estimate:Q", title="Income", format=",.0f"),
            alt.Tooltip("avg_response_time:Q", title="Avg Response (hours)", format=",.1f"),
        ],
    )

    points = base.mark_circle(size=90, opacity=0.82, stroke="white", strokeWidth=0.8)

    trend_line = (
        alt.Chart(clean)
        .transform_regression("income_estimate", "avg_response_time", method="linear")
        .mark_line(color="#222222", size=2)
        .encode(x="income_estimate:Q", y="avg_response_time:Q")
    )

    label_points = (
        alt.Chart(labels_df)
        .mark_text(dx=8, dy=-8, fontSize=11, color="#111111")
        .encode(
            x="income_estimate:Q",
            y="avg_response_time:Q",
            text=alt.Text("community_area_name:N"),
        )
    )

    chart = (
        (points + trend_line + label_points)
        .properties(
            width=850,
            height=520,
            title={
                "text": "Income vs Average 311 Response Time (Altair)",
                "subtitle": [
                    "Outliers removed with IQR rule",
                    f"Linear trend R² = {r2:.3f}",
                    "Labeled: 3 poorest and 3 richest community areas",
                ],
            },
        )
        .configure_axis(grid=True)
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(OUT_PATH)

    print(f"Saved plot: {OUT_PATH}")


if __name__ == "__main__":
    main()
