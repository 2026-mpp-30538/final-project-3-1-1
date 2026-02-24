from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


SCRIPT_DIR = Path(__file__).parent
DERIVED_DIR = SCRIPT_DIR / "../data/derived-data"
DF_311_PATH = DERIVED_DIR / "df_311_ca.csv"
ACS_PATH = DERIVED_DIR / "acs_filtered.csv"
PLOTS_DIR = DERIVED_DIR / "plots"


def ensure_input(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"Saved plot: {path}")


def load_data() -> pd.DataFrame:
    ensure_input(DF_311_PATH)
    ensure_input(ACS_PATH)

    df_311 = pd.read_csv(DF_311_PATH)
    acs = pd.read_csv(ACS_PATH)

    for col in [
        "community_area",
        "total_requests",
        "avg_response_time",
        "income_estimate",
        "total_population",
        "requests_per_1000",
    ]:
        if col in df_311.columns:
            df_311[col] = pd.to_numeric(df_311[col], errors="coerce")

    for col in [
        "community_area_number",
        "income_estimate",
        "pct_black",
        "pct_hispanic",
        "pct_white",
        "pct_asian",
        "total_population",
    ]:
        if col in acs.columns:
            acs[col] = pd.to_numeric(acs[col], errors="coerce")

    merged = df_311.merge(
        acs,
        left_on="community_area_name",
        right_on="community_area_name",
        how="left",
        suffixes=("", "_acs"),
    )

    if "community_area_number" not in merged.columns:
        merged["community_area_number"] = merged["community_area"]

    merged = merged.sort_values("community_area_number")
    return merged


def plot_choropleth_style(df: pd.DataFrame, out_path: Path) -> None:
    # Grid-based choropleth substitute using community area IDs and requests_per_1000.
    data = df[["community_area_number", "requests_per_1000"]].dropna().copy()
    data = data.sort_values("community_area_number").reset_index(drop=True)

    cols = 11
    rows = int(np.ceil(len(data) / cols))
    cmap = plt.cm.YlOrRd
    vmin = data["requests_per_1000"].min()
    vmax = data["requests_per_1000"].max()

    fig, ax = plt.subplots(figsize=(14, 8))
    for idx, row in data.iterrows():
        r = idx // cols
        c = idx % cols
        value = row["requests_per_1000"]
        norm = 0 if vmax == vmin else (value - vmin) / (vmax - vmin)
        color = cmap(norm)

        rect = Rectangle((c, rows - r - 1), 1, 1, facecolor=color, edgecolor="white", linewidth=1)
        ax.add_patch(rect)
        ax.text(c + 0.5, rows - r - 0.5, f"{int(row['community_area_number'])}", ha="center", va="center", fontsize=8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("Requests per 1,000 residents")

    ax.set_title("Choropleth-Style Grid: 311 Requests per 1,000 by Community Area", fontsize=13)
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    save_fig(out_path)


def plot_line_requests(df: pd.DataFrame, out_path: Path) -> None:
    line_df = df[["community_area_number", "total_requests", "avg_response_time"]].dropna()

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(line_df["community_area_number"], line_df["total_requests"], color="#0077b6", linewidth=2, label="Total Requests")
    ax1.set_xlabel("Community Area Number")
    ax1.set_ylabel("Total Requests", color="#0077b6")
    ax1.tick_params(axis="y", labelcolor="#0077b6")

    ax2 = ax1.twinx()
    ax2.plot(line_df["community_area_number"], line_df["avg_response_time"], color="#d62828", linewidth=2, label="Avg Response Time (hours)")
    ax2.set_ylabel("Avg Response Time (hours)", color="#d62828")
    ax2.tick_params(axis="y", labelcolor="#d62828")

    fig.suptitle("Lines: Request Volume and Response Time by Community Area", fontsize=13)
    ax1.grid(alpha=0.25)
    save_fig(out_path)


def plot_scatter_income_requests(df: pd.DataFrame, out_path: Path) -> None:
    scat_df = df[["income_estimate", "requests_per_1000", "community_area_name"]].dropna()

    plt.figure(figsize=(10, 6))
    plt.scatter(scat_df["income_estimate"], scat_df["requests_per_1000"], alpha=0.75, color="#2a9d8f", edgecolors="white", linewidth=0.5)

    if len(scat_df) > 1:
        coeffs = np.polyfit(scat_df["income_estimate"], scat_df["requests_per_1000"], 1)
        x = np.linspace(scat_df["income_estimate"].min(), scat_df["income_estimate"].max(), 100)
        y = coeffs[0] * x + coeffs[1]
        plt.plot(x, y, color="#264653", linewidth=2)

    plt.title("Scatter: Estimated Income vs 311 Requests per 1,000", fontsize=13)
    plt.xlabel("Estimated Household Income (USD)")
    plt.ylabel("Requests per 1,000 residents")
    plt.grid(alpha=0.25)
    save_fig(out_path)


def plot_scatter_demographics_response(df: pd.DataFrame, out_path: Path) -> None:
    dem_df = df[["pct_black", "avg_response_time"]].dropna()

    plt.figure(figsize=(10, 6))
    plt.scatter(dem_df["pct_black"], dem_df["avg_response_time"], alpha=0.75, color="#f4a261", edgecolors="white", linewidth=0.5)

    if len(dem_df) > 1:
        coeffs = np.polyfit(dem_df["pct_black"], dem_df["avg_response_time"], 1)
        x = np.linspace(dem_df["pct_black"].min(), dem_df["pct_black"].max(), 100)
        y = coeffs[0] * x + coeffs[1]
        plt.plot(x, y, color="#e76f51", linewidth=2)

    plt.title("Scatter: Percent Black Population vs Avg Response Time", fontsize=13)
    plt.xlabel("Black Population (%)")
    plt.ylabel("Average Response Time (hours)")
    plt.grid(alpha=0.25)
    save_fig(out_path)


def plot_bar_top_requests(df: pd.DataFrame, out_path: Path) -> None:
    top_df = df[["community_area_name", "total_requests"]].dropna().sort_values("total_requests", ascending=False).head(10)
    top_df = top_df.sort_values("total_requests", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(top_df["community_area_name"], top_df["total_requests"], color="#457b9d")
    plt.title("Top 10 Community Areas by Total 311 Requests", fontsize=13)
    plt.xlabel("Total Requests")
    plt.ylabel("Community Area")
    plt.grid(axis="x", alpha=0.25)
    save_fig(out_path)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    plot_choropleth_style(df, PLOTS_DIR / "plot_1_choropleth_style_requests_per_1000.png")
    plot_line_requests(df, PLOTS_DIR / "plot_2_lines_requests_response.png")
    plot_scatter_income_requests(df, PLOTS_DIR / "plot_3_scatter_income_requests_per_1000.png")
    plot_scatter_demographics_response(df, PLOTS_DIR / "plot_4_scatter_pctblack_response_time.png")
    plot_bar_top_requests(df, PLOTS_DIR / "plot_5_bar_top10_total_requests.png")


if __name__ == "__main__":
    main()
