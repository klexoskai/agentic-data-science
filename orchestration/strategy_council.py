"""
orchestration/strategy_council.py — LangGraph graph for the Strategy Council.

Full topology:

    enrichment
        ↓
    [council_data_scientist,          ← fan-out (sequential for simplicity,
     council_sales_director,             fan-out parallelism is a LangGraph Pro
     council_qa_engineer]                feature; sequential is equivalent here)
        ↓
    debate
        ↓
    synthesiser   ← interim synthesis so Critic round 1 has a clean doc
        ↓
    critic_1
        ↓
    synthesiser   ← re-synthesise incorporating round-1 critique
        ↓
    critic_2
        ↓
    synthesiser   ← final polished output
        ↓
    [END]

The synthesiser runs 3 times: once before each critic pass, and once final.
This ensures each critic round reviews a coherent document, not raw notes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph

from orchestration.strategy_state import StrategyState
from orchestration.strategy_nodes import (
    COUNCIL_MEMBER_KEYS,
    council_nodes,
    critic_node_1,
    critic_node_2,
    debate_node,
    enrichment_node,
    synthesiser_node,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_strategy_graph() -> StateGraph:
    """Build and compile the Strategy Council graph."""
    graph = StateGraph(StrategyState)

    # ── Nodes ──────────────────────────────────────────────────────────────

    graph.add_node("enrichment", enrichment_node)

    # Council — one node per member
    for key, node_fn in zip(COUNCIL_MEMBER_KEYS, council_nodes):
        graph.add_node(f"council_{key}", node_fn)

    graph.add_node("debate", debate_node)
    graph.add_node("synthesiser_pre_critique", synthesiser_node)
    graph.add_node("critic_1", critic_node_1)
    graph.add_node("synthesiser_post_critique_1", synthesiser_node)
    graph.add_node("critic_2", critic_node_2)
    graph.add_node("synthesiser_final", synthesiser_node)

    # ── Edges ──────────────────────────────────────────────────────────────

    # Entry → enrichment
    graph.set_entry_point("enrichment")

    # Enrichment → council members (sequential fan-out)
    graph.add_edge("enrichment", f"council_{COUNCIL_MEMBER_KEYS[0]}")
    for i in range(len(COUNCIL_MEMBER_KEYS) - 1):
        graph.add_edge(
            f"council_{COUNCIL_MEMBER_KEYS[i]}",
            f"council_{COUNCIL_MEMBER_KEYS[i + 1]}",
        )

    # Last council member → debate
    graph.add_edge(f"council_{COUNCIL_MEMBER_KEYS[-1]}", "debate")

    # Debate → first synthesis → critic 1 → re-synthesis → critic 2 → final
    graph.add_edge("debate", "synthesiser_pre_critique")
    graph.add_edge("synthesiser_pre_critique", "critic_1")
    graph.add_edge("critic_1", "synthesiser_post_critique_1")
    graph.add_edge("synthesiser_post_critique_1", "critic_2")
    graph.add_edge("critic_2", "synthesiser_final")
    graph.add_edge("synthesiser_final", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_strategy_council(
    problem_statement: str,
    dataset_descriptions: list[str],
    quality_preset: str = "balanced",
    run_id: str = "",
    output_path: Path | None = None,
) -> dict[str, Any]:
    """
    Run the full Strategy Council for a given business problem.

    Parameters
    ----------
    problem_statement : str
        The business problem to solve (freeform text or path to a .md file).
    dataset_descriptions : list[str]
        List of dataset identifiers or descriptions, e.g.
        ["copa.csv — historical actuals", "launch_tracker25_matched.csv"].
    quality_preset : str
        Preset name for logging/metadata ("fast", "balanced", "maximum").
    run_id : str
        Optional unique ID for this run (used in memory snapshot + output filename).
    output_path : Path | None
        If provided, writes the final strategy markdown to this path.

    Returns
    -------
    dict with:
        final_strategy: str     — the full strategy document
        critiques: list[str]    — both critique rounds
        debate_summary: str     — the moderated debate output
        proposals: dict         — individual agent proposals
        retrieved_chunks: list  — all context chunks used
    """
    graph = build_strategy_graph()

    initial_state = StrategyState(
        problem_statement=problem_statement,
        dataset_descriptions=dataset_descriptions,
        quality_preset=quality_preset,
        run_id=run_id,
    )

    logger.info("[StrategyCouncil] Starting run: %s", run_id or "unnamed")
    final_state = graph.invoke(initial_state.model_dump())

    result = {
        "final_strategy": final_state.get("final_strategy", ""),
        "critiques": final_state.get("critiques", []),
        "debate_summary": final_state.get("debate_summary", ""),
        "proposals": final_state.get("proposals", {}),
        "retrieved_chunks": final_state.get("retrieved_chunks", []),
    }

    # Optionally write the strategy doc to disk
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result["final_strategy"], encoding="utf-8")
        logger.info("[StrategyCouncil] Strategy written to: %s", output_path)

    return result
