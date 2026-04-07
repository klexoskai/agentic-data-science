"""Data profiling tool.

Reads CSV or Excel files from a samples directory and produces a structured
profile report covering column types, null counts, unique values,
distributions, and suspicious patterns.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _profile_column(series: pd.Series) -> dict[str, Any]:
    """Produce a profile dict for a single pandas Series."""
    profile: dict[str, Any] = {
        "dtype": str(series.dtype),
        "count": int(series.count()),
        "null_count": int(series.isna().sum()),
        "null_pct": round(series.isna().mean() * 100, 2),
        "unique_values": int(series.nunique()),
    }

    if pd.api.types.is_numeric_dtype(series):
        desc = series.describe()
        profile.update(
            {
                "mean": round(float(desc.get("mean", 0)), 4),
                "std": round(float(desc.get("std", 0)), 4),
                "min": float(desc.get("min", 0)),
                "25%": float(desc.get("25%", 0)),
                "50%": float(desc.get("50%", 0)),
                "75%": float(desc.get("75%", 0)),
                "max": float(desc.get("max", 0)),
            }
        )
        # Flag suspicious patterns
        if profile["null_pct"] > 30:
            profile["flag"] = "HIGH_NULL_RATE"
        if profile["unique_values"] == 1:
            profile["flag"] = "CONSTANT_COLUMN"
    else:
        # Categorical / string column
        top_values = series.value_counts().head(5).to_dict()
        profile["top_values"] = {str(k): int(v) for k, v in top_values.items()}
        if profile["unique_values"] == series.count():
            profile["flag"] = "POSSIBLY_UNIQUE_ID"
        if profile["null_pct"] > 30:
            profile["flag"] = "HIGH_NULL_RATE"

    return profile


def _detect_suspicious_headers(columns: list[str]) -> list[str]:
    """Flag column names that are ambiguous or unconventional."""
    suspicious: list[str] = []
    ambiguous_patterns = ["%", "pct", "amt", "val", "num", "id", "flag", "code"]
    for col in columns:
        col_lower = col.lower().strip()
        # Short or cryptic names
        if len(col_lower) <= 2:
            suspicious.append(f"'{col}' — very short name, meaning unclear")
        # Contains % which is ambiguous
        elif "%" in col:
            suspicious.append(
                f"'{col}' — contains '%', could be a percentage or absolute value"
            )
        # Generic names
        elif col_lower in ("value", "amount", "data", "info", "type", "status"):
            suspicious.append(f"'{col}' — generic name, consider clarifying")
    return suspicious


@tool
def profile_data(file_path: str, max_rows: int = 50000) -> str:
    """Profile a CSV or Excel data file and return a structured report.

    Use this tool to understand the shape, quality, and quirks of a dataset
    before making modelling or pipeline decisions.

    Args:
        file_path: Absolute or relative path to a CSV or Excel file.
        max_rows: Maximum rows to read for profiling (default 50 000).
                  Larger files are sampled.

    Returns:
        A JSON-formatted profile report including per-column statistics,
        suspicious patterns, and overall data quality summary.
    """
    logger.info("profile_data: profiling %s", file_path)

    path = Path(file_path)
    if not path.exists():
        return json.dumps({"error": f"File not found: {file_path}"})

    # Read data
    try:
        if path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(path, nrows=max_rows)
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path, nrows=max_rows)
        elif path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
            if len(df) > max_rows:
                df = df.sample(n=max_rows, random_state=42)
        else:
            return json.dumps({"error": f"Unsupported file format: {path.suffix}"})
    except Exception as exc:
        return json.dumps({"error": f"Failed to read file: {exc}"})

    # Build profile
    report: dict[str, Any] = {
        "file": str(path),
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": {},
        "suspicious_headers": _detect_suspicious_headers(list(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    for col in df.columns:
        report["columns"][col] = _profile_column(df[col])

    # Overall quality score
    total_cells = df.shape[0] * df.shape[1]
    total_nulls = int(df.isna().sum().sum())
    report["overall_quality"] = {
        "total_cells": total_cells,
        "total_nulls": total_nulls,
        "completeness_pct": round((1 - total_nulls / max(total_cells, 1)) * 100, 2),
    }

    logger.info(
        "profile_data: %s — %d rows × %d cols, %.1f%% complete",
        path.name,
        len(df),
        len(df.columns),
        report["overall_quality"]["completeness_pct"],
    )

    return json.dumps(report, indent=2, default=str)
