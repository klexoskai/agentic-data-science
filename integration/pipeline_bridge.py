"""
pipeline_bridge.py — Wires AIQ research graph into your existing SwarmState pipeline.

Pattern: replace the hard-coded "Phase A — Understand" prompt with a
research graph call, so agents receive a richer, citation-backed context
instead of raw markdown files.

Usage in your existing run.py:

    from integration.pipeline_bridge import enrich_context_with_research

    # After loading context_text / data_sources_text, before build_graph():
    enriched = enrich_context_with_research(context_text, data_sources_text)
    initial_state = SwarmState(
        context_text=enriched["enriched_context"],
        ...
    )
"""

from __future__ import annotations

import logging
from typing import Any

from orchestration.graph import run_research
from store.retriever import retrieve_for_query as _store_retrieve

logger = logging.getLogger(__name__)

# Questions the research graph will pre-answer before the swarm starts.
# Tune these to your FMCG domain — they mirror AI-Q's "pre-research" pass.
_PRE_RESEARCH_QUERIES = [
    "What are the key analytical questions given the business context and data sources?",
    "What data quality issues or gaps should the data science team be aware of?",
    "What market trends or benchmarks are relevant to the data provided?",
]


def enrich_context_with_research(
    context_text: str,
    data_sources_text: str,
    retrieved_chunks: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Run pre-research queries and inject findings back into context_text.

    This gives your Phase A agents (Data Scientist, Sales Director, QA)
    a head-start with structured, citation-backed insights — the same pattern
    AI-Q uses before routing to its shallow/deep agents.

    Parameters
    ----------
    context_text : str
        Raw business context markdown.
    data_sources_text : str
        Raw data sources markdown.
    retrieved_chunks : list[dict] | None
        Optional pre-retrieved vector DB chunks {"source": str, "text": str}.

    Returns
    -------
    dict with:
        enriched_context: str  — original context + research findings appended
        research_results: list[dict]  — raw results per query
        all_citations: list[str]      — deduplicated citations
    """
    research_results = []
    all_citations: list[str] = []

    for query in _PRE_RESEARCH_QUERIES:
        logger.info("[Bridge] Running pre-research: %r", query[:60])
        # Use store retriever if no chunks passed in
        chunks = retrieved_chunks
        if chunks is None:
            try:
                chunks = _store_retrieve(
                    query=query,
                    context_text=context_text,
                    data_sources_text=data_sources_text,
                )
            except Exception:
                chunks = []
        result = run_research(
            query=query,
            context_text=context_text,
            data_sources_text=data_sources_text,
            retrieved_chunks=chunks,
            max_deep_iterations=2,  # keep pre-research shallow/fast
        )
        research_results.append({"query": query, **result})
        all_citations.extend(result.get("citations", []))

    # Deduplicate citations
    all_citations = list(dict.fromkeys(all_citations))

    # Append research findings to context for the swarm agents
    findings_block = "\n\n".join(
        f"### Pre-Research: {r['query']}\n{r['answer']}"
        for r in research_results
        if r.get("answer")
    )

    enriched_context = (
        context_text
        + "\n\n---\n\n## Pre-Research Findings (AIQ Research Pass)\n\n"
        + findings_block
    )

    return {
        "enriched_context": enriched_context,
        "research_results": research_results,
        "all_citations": all_citations,
    }
