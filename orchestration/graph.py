"""
AIQ-style LangGraph graph wired to your existing agentic-data-science stack.

Graph topology:

                        ┌──────────────┐
                        │ orchestrator │  ← classifies intent + depth
                        └──────┬───────┘
                               │
              ┌────────────────┼─────────────────┐
              ▼                ▼                  ▼
           meta_node     shallow_node      deep_planner
                              │                  │
                              ▼                  ▼
                            [END]     deep_researcher (loop)
                                               │
                                    (iteration < max_iterations
                                     AND subtasks remaining?)
                                               │ yes → loop
                                               │ no  ▼
                                          synthesiser
                                               │
                                             [END]

Integration with your existing SwarmState pipeline:
  - The graph accepts a ResearchState that contains `context_text` and
    `data_sources_text` fields, mirroring SwarmState inputs.
  - You can call `run_research(query, context_text, data_sources_text)`
    and receive a structured answer + citations to feed into Gate 1 or
    directly into a build prompt.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from orchestration.state import QueryIntent, ResearchDepth, ResearchState
from orchestration.nodes import (
    deep_planner_node,
    deep_researcher_node,
    meta_node,
    orchestration_node,
    shallow_node,
    synthesiser_node,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def _route_after_orchestrator(state: ResearchState) -> str:
    """Route based on intent and depth classification."""
    if state.intent == QueryIntent.META:
        return "meta"
    if state.depth == ResearchDepth.DEEP:
        return "deep_planner"
    return "shallow"


def _should_continue_deep_research(state: ResearchState) -> str:
    """Loop deep_researcher until all subtasks covered or max_iterations hit."""
    if state.iteration >= state.max_iterations:
        logger.info("[DeepLoop] Max iterations reached — synthesising.")
        return "synthesise"
    if state.plan and state.iteration >= len(state.plan):
        logger.info("[DeepLoop] All subtasks covered — synthesising.")
        return "synthesise"
    logger.info("[DeepLoop] Continuing — iteration %d / %d.", state.iteration, state.max_iterations)
    return "continue"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_research_graph(max_deep_iterations: int = 3) -> StateGraph:
    """
    Build and compile the AIQ-style research graph.

    Parameters
    ----------
    max_deep_iterations : int
        Maximum number of deep research loop iterations before forcing synthesis.

    Returns
    -------
    Compiled StateGraph ready to invoke with a ResearchState dict.
    """
    graph = StateGraph(ResearchState)

    # ── Nodes ──────────────────────────────────────────────────────────────

    graph.add_node("orchestrator", orchestration_node)
    graph.add_node("meta", meta_node)
    graph.add_node("shallow", shallow_node)
    graph.add_node("deep_planner", deep_planner_node)
    graph.add_node("deep_researcher", deep_researcher_node)
    graph.add_node("synthesiser", synthesiser_node)

    # ── Entry ──────────────────────────────────────────────────────────────

    graph.set_entry_point("orchestrator")

    # ── Edges ──────────────────────────────────────────────────────────────

    # Orchestrator → intent/depth routing
    graph.add_conditional_edges(
        "orchestrator",
        _route_after_orchestrator,
        {
            "meta": "meta",
            "shallow": "shallow",
            "deep_planner": "deep_planner",
        },
    )

    # Meta and shallow are terminal
    graph.add_edge("meta", END)
    graph.add_edge("shallow", END)

    # Deep planner → first researcher iteration
    graph.add_edge("deep_planner", "deep_researcher")

    # Deep researcher loop
    graph.add_conditional_edges(
        "deep_researcher",
        _should_continue_deep_research,
        {
            "continue": "deep_researcher",
            "synthesise": "synthesiser",
        },
    )

    # Synthesiser is terminal
    graph.add_edge("synthesiser", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Convenience runner — drop-in for your existing run.py
# ---------------------------------------------------------------------------

def run_research(
    query: str,
    context_text: str = "",
    data_sources_text: str = "",
    max_deep_iterations: int = 3,
    retrieved_chunks: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Run a single query through the AIQ-style research graph.

    Parameters
    ----------
    query : str
        The user's question or research task.
    context_text : str
        Business context markdown (mirrors SwarmState.context_text).
    data_sources_text : str
        Data sources doc (mirrors SwarmState.data_sources_text).
    max_deep_iterations : int
        Deep research loop cap.
    retrieved_chunks : list[dict] | None
        Pre-retrieved context chunks. If None, the store.retriever is called
        automatically to fetch relevant chunks from ChromaDB.

    Returns
    -------
    dict with keys: answer, citations, intent, depth, tool_calls
    """
    # Use ChromaDB retriever if no chunks supplied
    if retrieved_chunks is None:
        try:
            from store.retriever import retrieve_for_query
            retrieved_chunks = retrieve_for_query(
                query=query,
                context_text=context_text,
                data_sources_text=data_sources_text,
            )
        except Exception:
            retrieved_chunks = []

    graph = build_research_graph(max_deep_iterations=max_deep_iterations)

    initial_state = ResearchState(
        query=query,
        context_text=context_text,
        data_sources_text=data_sources_text,
        retrieved_chunks=retrieved_chunks,
        max_iterations=max_deep_iterations,
    )

    final_state = graph.invoke(initial_state.model_dump())

    return {
        "answer": final_state.get("answer", ""),
        "citations": final_state.get("citations", []),
        "intent": final_state.get("intent"),
        "depth": final_state.get("depth"),
        "tool_calls": final_state.get("tool_calls", []),
    }
