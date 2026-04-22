"""
example_usage.py — Runnable examples showing how to use the AIQ research graph.

Run with:
    python example_usage.py
"""

from __future__ import annotations

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

from orchestration.graph import run_research


# ---------------------------------------------------------------------------
# Example 1: Shallow research — single metric question
# ---------------------------------------------------------------------------

def example_shallow():
    print("\n" + "="*60)
    print("EXAMPLE 1: Shallow research")
    print("="*60)
    print("NOTE: Run 'python -m store.ingest' first to populate ChromaDB.")

    # retrieved_chunks=None → automatically fetched from ChromaDB
    result = run_research(
        query="What is our YTD revenue in SEA and which market is growing fastest?",
    )

    print(f"\nIntent : {result['intent']}")
    print(f"Depth  : {result['depth']}")
    print(f"\nAnswer :\n{result['answer']}")
    print(f"\nCitations: {result['citations']}")


# ---------------------------------------------------------------------------
# Example 2: Deep research — multi-step analytical question
# ---------------------------------------------------------------------------

def example_deep():
    print("\n" + "="*60)
    print("EXAMPLE 2: Deep research")
    print("="*60)
    print("NOTE: Run 'python -m store.ingest' first to populate ChromaDB.")

    # retrieved_chunks=None → automatically fetched from ChromaDB
    result = run_research(
        query=(
            "Give me a comprehensive Q2 market analysis for our FMCG portfolio in SEA: "
            "revenue trends, competitive position, and top 3 risks for Q3."
        ),
        max_deep_iterations=3,
    )

    print(f"\nIntent : {result['intent']}")
    print(f"Depth  : {result['depth']}")
    print(f"\nReport :\n{result['answer']}")
    print(f"\nCitations: {result['citations']}")
    print(f"\nTool call trace ({len(result['tool_calls'])} calls recorded).")


# ---------------------------------------------------------------------------
# Example 3: Meta query
# ---------------------------------------------------------------------------

def example_meta():
    print("\n" + "="*60)
    print("EXAMPLE 3: Meta query")
    print("="*60)

    result = run_research(query="What can this research agent help me with?")
    print(f"\nIntent : {result['intent']}")
    print(f"\nAnswer :\n{result['answer']}")


# ---------------------------------------------------------------------------
# Example 4: Bridge into your existing SwarmState pipeline
# ---------------------------------------------------------------------------

def example_pipeline_bridge():
    print("\n" + "="*60)
    print("EXAMPLE 4: Pipeline bridge (enriches SwarmState context)")
    print("="*60)

    from integration.pipeline_bridge import enrich_context_with_research

    context_text = """
# Project: SEA FMCG Forecasting
## Objective
Build a 12-month sales forecast for the OTC cough/cold portfolio across SEA markets.
Key deliverable: market-level volume projections with confidence intervals.
"""
    data_sources_text = """
- pnl2425_volume_extracts.csv: P&L and volume data for 2024-2025
- launch_tracker25.csv: New SKU launch history
- IQVIA_Asia_data1.csv: Third-party market size data
"""
    stub_chunks = [
        {"source": "pnl2425_volume_extracts.csv", "text": "SEA YTD revenue: $42.3M (+8% YoY)."},
    ]

    enriched = enrich_context_with_research(
        context_text=context_text,
        data_sources_text=data_sources_text,
        retrieved_chunks=stub_chunks,
    )

    print(f"\nEnriched context length: {len(enriched['enriched_context'])} chars")
    print(f"Citations collected: {len(enriched['all_citations'])}")
    print("\n--- Enriched context preview (first 800 chars) ---")
    print(enriched["enriched_context"][:800])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    example_shallow()
    example_meta()
    example_deep()
    example_pipeline_bridge()
