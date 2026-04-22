"""
AIQ-style orchestration state for agentic-data-science.

Extends the existing SwarmState with AI-Q concepts:
- intent classification (meta vs. research)
- research depth routing (shallow vs. deep)
- citation tracking
- tool call trace for observability
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ResearchDepth(str, Enum):
    SHALLOW = "shallow"   # bounded: fast answer + citations, ~5 tool calls
    DEEP = "deep"         # multi-phase: plan → iterate → synthesise → report


class QueryIntent(str, Enum):
    META = "meta"         # "what can you do?", "rephrase this" — no data tools needed
    RESEARCH = "research" # requires data retrieval + reasoning


# ---------------------------------------------------------------------------
# Reducer helpers (reuse pattern from your existing swarm.py)
# ---------------------------------------------------------------------------

def _merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}

def _merge_lists(a: list, b: list) -> list:
    return a + b


# ---------------------------------------------------------------------------
# ResearchState — the new top-level state for AIQ-style routing
# ---------------------------------------------------------------------------

class ResearchState(BaseModel):
    """State tracked across the full AIQ-style research graph."""

    # --- Query ---
    query: str = ""
    intent: QueryIntent = QueryIntent.RESEARCH
    depth: ResearchDepth = ResearchDepth.SHALLOW

    # --- Orchestration ---
    plan: list[str] = Field(default_factory=list)          # deep agent's subtask plan
    iteration: int = 0
    max_iterations: int = 3                                  # deep loop cap

    # --- Retrieved context ---
    retrieved_chunks: Annotated[list[dict], _merge_lists] = Field(default_factory=list)
    tool_calls: Annotated[list[dict], _merge_lists] = Field(default_factory=list)

    # --- Agent outputs ---
    agent_outputs: Annotated[dict[str, Any], _merge_dicts] = Field(default_factory=dict)

    # --- Final deliverable ---
    answer: str = ""
    citations: list[str] = Field(default_factory=list)

    # --- Pass-through to existing SwarmState fields (for wiring into your pipeline) ---
    context_text: str = ""          # business context (from SwarmState)
    data_sources_text: str = ""     # data sources doc (from SwarmState)

    class Config:
        arbitrary_types_allowed = True
