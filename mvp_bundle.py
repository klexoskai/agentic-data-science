"""MVP launch projection bundle generator.

Creates deterministic output files from local data and powers the Dash frontend.
The logic is intentionally simple for first-run usability:
- infer top similar historical SKUs from launch attributes
- build a 12-month projection from historical TM1 curves
- export CSV/HTML/Markdown artefacts
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go


@dataclass
class MVPInput:
    """Hypothetical launch input used for similarity and projection."""

    launch_month: int
    category: str
    market: str
    brand: str


def _safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, engine="python")


def _clean_numeric(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .replace({"": None, "nan": None, "None": None})
        .astype(float)
    )


def _to_month(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.to_period("M").dt.to_timestamp()


def _month_num(value: Any) -> int | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return int(ts.month)


def _prepare_launch_tracker(data_dir: Path) -> pd.DataFrame:
    df = _safe_read_csv(data_dir / "launch_tracker25_matched.csv")
    expected = {
        "SKU Code": "sku_code",
        "SKU Name": "sku_name",
        "SKU Launch Month": "launch_month_raw",
        "Category": "category",
        "Market Specific": "market",
        "Brand": "brand",
    }
    cols = {k: v for k, v in expected.items() if k in df.columns}
    lt = df[list(cols.keys())].rename(columns=cols).copy()
    lt["sku_code"] = lt["sku_code"].astype(str).str.strip()
    lt["category"] = lt["category"].astype(str).str.strip()
    lt["market"] = lt["market"].astype(str).str.strip()
    lt["brand"] = lt["brand"].astype(str).str.strip()
    lt["launch_month_num"] = lt["launch_month_raw"].apply(_month_num)
    lt = lt.dropna(subset=["sku_code"]).drop_duplicates(subset=["sku_code"])
    return lt


def _prepare_tm1(data_dir: Path) -> pd.DataFrame:
    tm1 = _safe_read_csv(data_dir / "tm1_qty_sales_pivot.csv")
    if "sku_id" not in tm1.columns or "period" not in tm1.columns:
        raise ValueError("tm1_qty_sales_pivot.csv missing required columns: sku_id, period")

    tm1 = tm1.copy()
    tm1["sku_id"] = tm1["sku_id"].astype(str).str.strip()
    tm1["period"] = _to_month(tm1["period"])
    tm1["quantity_clean"] = _clean_numeric(tm1.get("quantity", pd.Series([0] * len(tm1))))
    tm1["net_sales_clean"] = _clean_numeric(tm1.get("net_sales", pd.Series([0] * len(tm1))))
    tm1 = tm1.dropna(subset=["sku_id", "period"])
    return tm1


def _build_curves(tm1: pd.DataFrame) -> pd.DataFrame:
    curves = tm1[["sku_id", "period", "quantity_clean", "net_sales_clean"]].copy()
    curves = curves.sort_values(["sku_id", "period"])
    first_period = curves.groupby("sku_id")["period"].transform("min")
    curves["month_idx"] = (
        (curves["period"].dt.year - first_period.dt.year) * 12
        + (curves["period"].dt.month - first_period.dt.month)
        + 1
    )
    curves = curves[(curves["month_idx"] >= 1) & (curves["month_idx"] <= 12)]
    return curves


def _score_similarity(candidates: pd.DataFrame, user_input: MVPInput) -> pd.DataFrame:
    df = candidates.copy()
    df["score"] = 0.0
    df["score"] += (df["category"].str.lower() == user_input.category.lower()).astype(float) * 3.0
    df["score"] += (df["market"].str.lower() == user_input.market.lower()).astype(float) * 3.0
    df["score"] += (df["brand"].str.lower() == user_input.brand.lower()).astype(float) * 2.0

    month_delta = (df["launch_month_num"] - user_input.launch_month).abs()
    df["score"] += month_delta.fillna(12).map(lambda x: max(0.0, 2.0 - (float(x) / 6.0)))
    return df.sort_values(["score"], ascending=False)


def generate_projection_bundle(
    data_dir: str | Path,
    output_dir: str | Path,
    user_input: MVPInput | None = None,
) -> dict[str, Path]:
    """Generate projection artefacts and return paths."""
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    lt = _prepare_launch_tracker(data_path)
    tm1 = _prepare_tm1(data_path)
    curves = _build_curves(tm1)

    if lt.empty:
        raise ValueError("Launch tracker has no usable rows.")

    if user_input is None:
        seed = lt.iloc[0]
        user_input = MVPInput(
            launch_month=int(seed.get("launch_month_num") or 1),
            category=str(seed.get("category") or ""),
            market=str(seed.get("market") or ""),
            brand=str(seed.get("brand") or ""),
        )

    ranked = _score_similarity(lt, user_input)
    ranked = ranked[ranked["sku_code"].isin(curves["sku_id"].unique())]
    similar = ranked.head(3).copy()
    if similar.empty:
        raise ValueError("No overlapping SKUs found between launch tracker and TM1 data.")

    overlay = curves[curves["sku_id"].isin(similar["sku_code"])].copy()
    projection = (
        overlay.groupby("month_idx", as_index=False)["quantity_clean"]
        .median()
        .rename(columns={"quantity_clean": "projected_quantity"})
    )
    projection = projection.sort_values("month_idx")

    fig = go.Figure()
    for sku in similar["sku_code"]:
        sku_curve = overlay[overlay["sku_id"] == sku].sort_values("month_idx")
        fig.add_trace(
            go.Scatter(
                x=sku_curve["month_idx"],
                y=sku_curve["quantity_clean"],
                mode="lines+markers",
                name=f"Similar SKU {sku}",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=projection["month_idx"],
            y=projection["projected_quantity"],
            mode="lines+markers",
            name="Projected (median of similar SKUs)",
            line={"width": 4},
        )
    )
    fig.update_layout(
        title="12-Month Projection with Similar SKU Overlays",
        xaxis_title="Months Since Launch",
        yaxis_title="Quantity",
        template="plotly_white",
    )

    similar_path = out_path / "similar_skus.csv"
    overlay_path = out_path / "similar_sku_overlay.csv"
    projection_path = out_path / "projected_curve.csv"
    chart_path = out_path / "projection_chart.html"
    report_path = out_path / "final_report.md"

    similar.to_csv(similar_path, index=False)
    overlay.to_csv(overlay_path, index=False)
    projection.to_csv(projection_path, index=False)
    fig.write_html(chart_path, include_plotlyjs="cdn")

    report_text = (
        "# MVP Projection Report\n\n"
        "## Input Scenario\n"
        f"- Launch month: `{user_input.launch_month}`\n"
        f"- Category: `{user_input.category}`\n"
        f"- Market: `{user_input.market}`\n"
        f"- Brand: `{user_input.brand}`\n\n"
        "## Similar SKUs (Top 3)\n"
        f"- {', '.join(similar['sku_code'].astype(str).tolist())}\n\n"
        "## Projection Method\n"
        "- Similarity scoring uses category, market, brand, and launch-month proximity.\n"
        "- 12-month projection is the median monthly quantity across selected similar SKUs.\n"
        "- Overlay curves are included to show comparable historical trajectories.\n"
    )
    report_path.write_text(report_text, encoding="utf-8")

    return {
        "similar_skus": similar_path,
        "overlay": overlay_path,
        "projection": projection_path,
        "chart": chart_path,
        "report": report_path,
    }

