"""
orchestration/strategy_state.py — State definition for the Strategy Council graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any
from pydantic import BaseModel, Field


def _merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}

def _merge_lists(a: list, b: list) -> list:
    return a + b


@dataclass
class CouncilMember:
    """Descriptor for a strategy council agent."""
    key: str
    role: str
    domain: str


class StrategyState(BaseModel):
    """Full state tracked across the Strategy Council graph execution."""

    # --- Inputs ---
    problem_statement: str = ""
    dataset_descriptions: list[str] = Field(default_factory=list)

    # --- Enrichment ---
    retrieved_chunks: Annotated[list[dict], _merge_lists] = Field(default_factory=list)

    # --- Council proposals (agent_key → proposal text) ---
    proposals: Annotated[dict[str, str], _merge_dicts] = Field(default_factory=dict)

    # --- Debate output ---
    debate_summary: str = ""

    # --- Critic outputs (one entry per round) ---
    critiques: Annotated[list[str], _merge_lists] = Field(default_factory=list)

    # --- Intermediate synthesised strategy (updated between critic rounds) ---
    final_strategy: str = ""

    # --- Metadata ---
    quality_preset: str = "balanced"
    run_id: str = ""

    class Config:
        arbitrary_types_allowed = True
