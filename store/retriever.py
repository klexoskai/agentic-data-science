"""
store/retriever.py — Query interface for all agent nodes.

This is the single import point that agent nodes (shallow_node, deep_researcher_node,
pipeline_bridge, etc.) use to fetch context from ChromaDB.

Usage:
    from store.retriever import retrieve, retrieve_for_query

    # General retrieval across all collections
    chunks = retrieve("SEA revenue trend Q2 2025", n_results=5)

    # Targeted retrieval from a specific collection
    chunks = retrieve("Betadine launch Australia", collection="data_sources", n_results=3)

    # Full-pipeline context fetch (used by pipeline_bridge)
    chunks = retrieve_for_query(query, context_text, data_sources_text)
"""

from __future__ import annotations

import logging
from typing import Literal

from store.client import get_collection
from store.config import (
    COLLECTION_AGENT_MEMORY,
    COLLECTION_CONTEXT_DOCS,
    COLLECTION_DATA_SOURCES,
    DEFAULT_N_RESULTS,
)

logger = logging.getLogger(__name__)

CollectionName = Literal["data_sources", "context_docs", "agent_memory", "all"]


# ── Core retrieval ────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    collection: CollectionName = "all",
    n_results: int = DEFAULT_N_RESULTS,
    where: dict | None = None,
) -> list[dict]:
    """
    Retrieve the most relevant chunks for a query.

    Parameters
    ----------
    query : str
        Natural language query.
    collection : str
        Which collection to search. "all" searches data_sources + context_docs
        and merges results by relevance score.
    n_results : int
        Number of chunks to return.
    where : dict | None
        Optional ChromaDB metadata filter, e.g. {"source": "copa.csv"}.

    Returns
    -------
    list[dict]  — each dict has keys: text, source, metadata, distance
    """
    if collection == "all":
        return _retrieve_all(query, n_results=n_results, where=where)

    col_name = {
        "data_sources": COLLECTION_DATA_SOURCES,
        "context_docs": COLLECTION_CONTEXT_DOCS,
        "agent_memory": COLLECTION_AGENT_MEMORY,
    }[collection]

    return _query_collection(col_name, query, n_results=n_results, where=where)


def _query_collection(
    col_name: str,
    query: str,
    n_results: int,
    where: dict | None,
) -> list[dict]:
    """Run a query against a single collection, returning normalised chunks."""
    try:
        col = get_collection(col_name)
    except Exception as exc:
        logger.warning("Collection '%s' not available: %s", col_name, exc)
        return []

    kwargs: dict = {"query_texts": [query], "n_results": n_results}
    if where:
        kwargs["where"] = where

    try:
        results = col.query(**kwargs)
    except Exception as exc:
        logger.warning("Query failed on '%s': %s", col_name, exc)
        return []

    chunks = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, distances):
        chunks.append({
            "text": doc,
            "source": meta.get("source", col_name),
            "metadata": meta,
            "distance": dist,
        })

    return chunks


def _retrieve_all(query: str, n_results: int, where: dict | None) -> list[dict]:
    """Search data_sources + context_docs, merge and re-rank by distance."""
    results = []
    per_col = max(n_results, 5)

    for col in [COLLECTION_DATA_SOURCES, COLLECTION_CONTEXT_DOCS]:
        results.extend(_query_collection(col, query, n_results=per_col, where=where))

    # Sort by distance (lower = more similar) and return top n
    results.sort(key=lambda x: x.get("distance", 1.0))
    return results[:n_results]


# ── Filtered retrieval helpers ────────────────────────────────────────────────

def retrieve_by_sku(sku_id: str | int, n_results: int = 10) -> list[dict]:
    """Retrieve all chunks related to a specific SKU ID."""
    return retrieve(
        query=f"SKU {sku_id} sales revenue forecast",
        collection="data_sources",
        n_results=n_results,
        where={"sku": str(sku_id)},
    )


def retrieve_context_docs(topic: str, n_results: int = 5) -> list[dict]:
    """Retrieve relevant sections from markdown context documents."""
    return retrieve(query=topic, collection="context_docs", n_results=n_results)


def retrieve_agent_memory(topic: str, n_results: int = 3) -> list[dict]:
    """Retrieve relevant past agent run outputs."""
    return retrieve(query=topic, collection="agent_memory", n_results=n_results)


# ── Pipeline-level retrieval (used by pipeline_bridge + AIQ nodes) ────────────

def retrieve_for_query(
    query: str,
    context_text: str = "",
    data_sources_text: str = "",
    n_results: int = DEFAULT_N_RESULTS,
) -> list[dict]:
    """
    Full-pipeline retrieval: combines semantic search with any context
    passed directly (for cases where docs aren't yet in the store).

    This is the function called by orchestration/nodes.py and
    integration/pipeline_bridge.py — replacing the stub `retrieved_chunks`.

    Parameters
    ----------
    query : str
        The agent's research query.
    context_text : str
        Raw context.md text (passed through as a synthetic chunk if non-empty).
    data_sources_text : str
        Raw data_sources.md text (passed through as a synthetic chunk).
    n_results : int
        Number of DB chunks to retrieve.

    Returns
    -------
    list[dict]  — {"text": str, "source": str, "metadata": dict, "distance": float}
    """
    chunks = retrieve(query=query, collection="all", n_results=n_results)

    # Prepend any raw context passed directly (synthetic chunks, distance=0)
    synthetic = []
    if context_text:
        synthetic.append({
            "text": context_text,
            "source": "context.md (runtime)",
            "metadata": {"source": "context.md"},
            "distance": 0.0,
        })
    if data_sources_text:
        synthetic.append({
            "text": data_sources_text,
            "source": "data_sources.md (runtime)",
            "metadata": {"source": "data_sources.md"},
            "distance": 0.0,
        })

    return synthetic + chunks
