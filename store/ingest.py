"""
store/ingest.py — Ingestion pipeline for all context sources.

Handles three ingestion strategies:

1. CSV data sources  → chunked by SKU + period window
   Sources: copa.csv, tm1_qty_sales_pivot.csv, pnl2425_volume_extracts_matched.csv,
            launch_tracker25_matched.csv, euro_mon_hier1_RSP*.csv, IQVIA_Asia_data1.csv

2. Markdown context docs → chunked by heading (## sections)
   Sources: inputs/sample/context.md, inputs/sample/data_sources.md,
            any *.md under inputs/, decisions/

3. Agent memory snapshots → written by pipeline at end of each run
   (See store/memory.py for the write side)

Usage:
    python -m store.ingest                        # ingest everything
    python -m store.ingest --source csv           # only CSV data
    python -m store.ingest --source docs          # only markdown docs
    python -m store.ingest --reset                # wipe + re-ingest
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import textwrap
from pathlib import Path
from typing import Any

import pandas as pd

from store.client import get_collection
from store.config import (
    COLLECTION_CONTEXT_DOCS,
    COLLECTION_DATA_SOURCES,
    DATA_DIR,
    INPUTS_DIR,
    MAX_CHUNK_CHARS,
    PROJECT_ROOT,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s — %(message)s")


# ── Helpers ─────────────────────────────────────────────────────────────────

def _doc_id(*parts: str) -> str:
    """Stable, collision-resistant document ID from arbitrary string parts."""
    key = "|".join(parts)
    return hashlib.md5(key.encode()).hexdigest()


def _chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split long text into overlapping chunks of max_chars."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    overlap = max_chars // 5
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start += max_chars - overlap
    return chunks


# ── 1. CSV ingestion ─────────────────────────────────────────────────────────

# Describes how to chunk and summarise each CSV file.
# "sku_col": column to group by for per-SKU chunks (None = no SKU grouping)
# "period_col": date/period column for labelling (None = omit)
# "value_cols": columns to include in the text summary
CSV_CONFIGS: list[dict[str, Any]] = [
    {
        "file": "launch_tracker25_matched.csv",
        "label": "Launch Tracker",
        "sku_col": "SKU Code",
        "period_col": "SKU Launch Month",
        "value_cols": ["SKU Code", "SKU Launch Month", "Category", "Market Specific", "Brand"],
    },
    {
        "file": "pnl2425_volume_extracts_matched.csv",
        "label": "P&L Actuals FY24-25",
        "sku_col": "matched_SKU_ID",
        "period_col": None,
        "value_cols": [
            "matched_SKU_ID", "project_name", "Market",
            "forecast_volume_y1", "forecast_net_sales_y1",
            "forecast_volume_y2", "forecast_net_sales_y2",
        ],
    },
    {
        "file": "tm1_qty_sales_pivot.csv",
        "label": "TM1 Sales Pivot",
        "sku_col": "sku_id",
        "period_col": "period",
        "value_cols": ["sku_id", "period", "quantity", "net_sales"],
    },
    {
        "file": "euro_mon_hier1_RSP_USD_histconst2024_histfixedER20242.csv",
        "label": "Euromonitor RSP",
        "sku_col": None,
        "period_col": None,
        "value_cols": None,   # include all columns
    },
    {
        "file": "IQVIA_Asia_data1.csv",
        "label": "IQVIA Asia",
        "sku_col": None,
        "period_col": None,
        "value_cols": None,
    },
]

# Large files chunked in row batches rather than per-SKU
LARGE_FILE_ROW_BATCH = 100


def _df_to_text(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    """Convert a DataFrame (or subset of columns) to a readable text block."""
    if cols:
        existing = [c for c in cols if c in df.columns]
        df = df[existing]
    return df.to_string(index=False, max_rows=200)


def ingest_csv_sources(reset: bool = False) -> None:
    """Ingest all CSV data files into the data_sources collection."""
    collection = get_collection(COLLECTION_DATA_SOURCES)

    if reset:
        logger.info("Resetting data_sources collection…")
        from store.client import get_client
        get_client().delete_collection(COLLECTION_DATA_SOURCES)
        collection = get_collection(COLLECTION_DATA_SOURCES)

    for cfg in CSV_CONFIGS:
        path = DATA_DIR / cfg["file"]
        if not path.exists():
            logger.warning("File not found, skipping: %s", path)
            continue

        logger.info("Ingesting %s (%s)…", cfg["file"], cfg["label"])

        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            logger.error("Failed to read %s: %s", cfg["file"], exc)
            continue

        # Normalise column names (strip whitespace)
        df.columns = [str(c).strip() for c in df.columns]

        sku_col = cfg.get("sku_col")
        value_cols = cfg.get("value_cols")
        label = cfg["label"]

        if sku_col and sku_col in df.columns:
            # Group by SKU — one chunk per SKU (or batched if large)
            groups = df.groupby(sku_col)
            for sku_val, group_df in groups:
                text = f"[{label}] SKU: {sku_val}\n" + _df_to_text(group_df, value_cols)
                for i, chunk in enumerate(_chunk_text(text)):
                    doc_id = _doc_id(cfg["file"], str(sku_val), str(i))
                    collection.upsert(
                        ids=[doc_id],
                        documents=[chunk],
                        metadatas=[{
                            "source": cfg["file"],
                            "label": label,
                            "sku": str(sku_val),
                            "chunk_index": i,
                        }],
                    )
        else:
            # No SKU col — chunk by row batches
            for batch_start in range(0, len(df), LARGE_FILE_ROW_BATCH):
                batch = df.iloc[batch_start: batch_start + LARGE_FILE_ROW_BATCH]
                text = f"[{label}] Rows {batch_start}–{batch_start + len(batch)}\n" + _df_to_text(batch, value_cols)
                for i, chunk in enumerate(_chunk_text(text)):
                    doc_id = _doc_id(cfg["file"], str(batch_start), str(i))
                    collection.upsert(
                        ids=[doc_id],
                        documents=[chunk],
                        metadatas=[{
                            "source": cfg["file"],
                            "label": label,
                            "batch_start": batch_start,
                            "chunk_index": i,
                        }],
                    )

        logger.info("  ✓ %s — %d rows ingested", cfg["file"], len(df))

    logger.info("CSV ingestion complete.")


# ── 2. Markdown context doc ingestion ────────────────────────────────────────

def _split_markdown_by_heading(text: str, source: str) -> list[dict]:
    """
    Split a markdown document by ## headings.
    Returns list of {"text": str, "heading": str, "source": str}.
    """
    import re
    sections = re.split(r'\n(?=##+ )', text.strip())
    chunks = []
    for section in sections:
        if not section.strip():
            continue
        # Extract heading
        first_line = section.split('\n', 1)[0].strip()
        heading = first_line.lstrip('#').strip() or "intro"
        # Further split if section is too long
        for i, chunk in enumerate(_chunk_text(section)):
            chunks.append({
                "text": chunk,
                "heading": heading,
                "source": source,
                "chunk_index": i,
            })
    return chunks


def ingest_context_docs(reset: bool = False) -> None:
    """Ingest all markdown context documents into the context_docs collection."""
    collection = get_collection(COLLECTION_CONTEXT_DOCS)

    if reset:
        logger.info("Resetting context_docs collection…")
        from store.client import get_client
        get_client().delete_collection(COLLECTION_CONTEXT_DOCS)
        collection = get_collection(COLLECTION_CONTEXT_DOCS)

    # Collect all markdown files from inputs/ and decisions/
    md_files: list[Path] = []
    for pattern in ["inputs/**/*.md", "decisions/*.md", "README.md"]:
        md_files.extend(PROJECT_ROOT.glob(pattern))

    # Also ingest the AIQ orchestration module context files if present
    for extra in [
        PROJECT_ROOT / "store" / "README.md",
    ]:
        if extra.exists():
            md_files.append(extra)

    if not md_files:
        logger.warning("No markdown files found to ingest.")
        return

    for md_path in md_files:
        rel_path = str(md_path.relative_to(PROJECT_ROOT))
        logger.info("Ingesting doc: %s", rel_path)

        text = md_path.read_text(encoding="utf-8")
        chunks = _split_markdown_by_heading(text, source=rel_path)

        for chunk in chunks:
            doc_id = _doc_id(rel_path, chunk["heading"], str(chunk["chunk_index"]))
            collection.upsert(
                ids=[doc_id],
                documents=[chunk["text"]],
                metadatas=[{
                    "source": rel_path,
                    "heading": chunk["heading"],
                    "chunk_index": chunk["chunk_index"],
                }],
            )

        logger.info("  ✓ %s — %d chunks", rel_path, len(chunks))

    logger.info("Context doc ingestion complete.")


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest context sources into ChromaDB.")
    parser.add_argument(
        "--source",
        choices=["csv", "docs", "all"],
        default="all",
        help="Which source type to ingest (default: all).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe existing collection data before ingesting.",
    )
    args = parser.parse_args()

    if args.source in ("csv", "all"):
        ingest_csv_sources(reset=args.reset)
    if args.source in ("docs", "all"):
        ingest_context_docs(reset=args.reset)


if __name__ == "__main__":
    main()
