"""LangGraph state machine — the core orchestration engine.

Defines the :class:`SwarmState` and builds a :class:`StateGraph` that
coordinates multiple agent personas through a phased pipeline:

    Phase A  (understand)  → Gate 1 → Phase B (build) → Phase C (test loop)
    → Gate 2 → Phase D (ship)

Each phase contains agent collaboration loops with self-reflection, peer
review, and conditional branching.
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from agents.personas.base import BasePersona
from gates.review import HumanReviewGate, ReviewVerdict

logger = logging.getLogger(__name__)


# ======================================================================
# State definition
# ======================================================================

class Phase(str, Enum):
    """Pipeline phases."""

    UNDERSTAND = "phase_a_understand"
    GATE_1 = "gate_1"
    BUILD = "phase_b_build"
    TEST = "phase_c_test"
    GATE_2 = "gate_2"
    SHIP = "phase_d_ship"
    DONE = "done"
    ABORTED = "aborted"


def _merge_dicts(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Merge two dicts, with b overriding a."""
    merged = {**a}
    merged.update(b)
    return merged


def _merge_lists(a: list[Any], b: list[Any]) -> list[Any]:
    """Append-only merge for lists."""
    return a + b


class SwarmState(BaseModel):
    """Full state tracked across the LangGraph execution.

    Uses Annotated fields with reducer functions so that LangGraph can
    merge incremental updates from each node.
    """

    # --- Phase tracking ---
    current_phase: Phase = Phase.UNDERSTAND
    review_count: int = 0

    # --- Inputs ---
    context_text: str = ""
    data_sources_text: str = ""
    samples_dir: str = ""

    # --- Agent outputs (agent_name → latest output) ---
    agent_outputs: Annotated[dict[str, Any], _merge_dicts] = Field(default_factory=dict)

    # --- Satisfaction flags (agent_name → bool) ---
    is_satisfied: Annotated[dict[str, bool], _merge_dicts] = Field(default_factory=dict)

    # --- Artefacts produced ---
    artifacts: Annotated[list[str], _merge_lists] = Field(default_factory=list)

    # --- Feedback log ---
    feedback_log: Annotated[list[str], _merge_lists] = Field(default_factory=list)

    # --- Workflow config ---
    max_review_loops: int = 5
    require_unanimous: bool = True
    reflection_enabled: bool = True
    best_practice_check_frequency: str = "on_uncertainty"

    class Config:
        arbitrary_types_allowed = True


# ======================================================================
# Node functions
# ======================================================================

def _make_understand_node(persona: BasePersona):
    """Create a Phase A node: persona analyses context and produces insights."""

    def node(state: SwarmState) -> dict[str, Any]:
        logger.info("[Phase A] %s is analysing business context…", persona.name)

        prompt = (
            "## Task: Understand the Business Context\n\n"
            "You are in **Phase A — Understanding**.  Read the business context "
            "and data source documentation below, then produce:\n"
            "1. **Key Findings** — What are the core analytical questions?\n"
            "2. **Proposed Approach** — How would you tackle this from your "
            "expert perspective?\n"
            "3. **Assumptions** — What assumptions are you making?\n"
            "4. **Risks & Concerns** — What could go wrong?\n"
            "5. **Data Quality Notes** — Any issues spotted in the data sources?\n\n"
            f"## Business Context\n{state.context_text}\n\n"
            f"## Data Sources\n{state.data_sources_text}\n"
        )

        # Include any prior feedback
        if state.feedback_log:
            prompt += (
                "\n\n## Previous Feedback (incorporate this):\n"
                + "\n".join(f"- {fb}" for fb in state.feedback_log)
            )

        output = persona.invoke(prompt)

        # Self-reflection if enabled
        if state.reflection_enabled:
            reflection = persona.reflect(output, prompt)
            if not persona.is_satisfied(reflection):
                logger.info("[Phase A] %s revising after self-reflection", persona.name)
                revision_prompt = (
                    f"Your self-reflection identified improvements:\n{reflection}\n\n"
                    f"Original output:\n{output}\n\n"
                    "Produce an improved version incorporating the feedback."
                )
                output = persona.invoke(revision_prompt)

        return {
            "agent_outputs": {persona.name: output},
            "is_satisfied": {persona.name: True},
        }

    return node


def _make_review_node(reviewer: BasePersona, reviewed_personas: list[BasePersona]):
    """Create a peer-review node where one persona reviews others' outputs."""

    def node(state: SwarmState) -> dict[str, Any]:
        all_satisfied = True
        reviews: list[str] = []

        for other in reviewed_personas:
            if other.name == reviewer.name:
                continue
            other_output = state.agent_outputs.get(other.name, "")
            if not other_output:
                continue

            logger.info("[Review] %s is reviewing %s's output", reviewer.name, other.name)
            review = reviewer.review(
                other_output=other_output,
                other_name=other.name,
                task_context=state.context_text,
            )
            reviews.append(f"**{reviewer.name} → {other.name}:** {review}")

            if not reviewer.is_satisfied(review):
                all_satisfied = False

        review_text = "\n\n".join(reviews)
        return {
            "agent_outputs": {f"{reviewer.name}_review": review_text},
            "is_satisfied": {reviewer.name: all_satisfied},
        }

    return node


def _make_build_node(persona: BasePersona):
    """Create a Phase B node: persona generates pipeline code / artefacts."""

    def node(state: SwarmState) -> dict[str, Any]:
        logger.info("[Phase B] %s is building pipeline components…", persona.name)

        # Collect all Phase A outputs for context
        phase_a_context = "\n\n---\n\n".join(
            f"### {name}\n{output}"
            for name, output in state.agent_outputs.items()
            if not name.endswith("_review")
        )

        prompt = (
            "## Task: Build Pipeline Components\n\n"
            "You are in **Phase B — Build**.  Based on the agreed architecture "
            "from Phase A (below), generate the concrete pipeline components "
            "from your expert perspective.\n\n"
            "Use the `generate_code` tool to create Python files.\n"
            "Use the `generate_diagram` tool to create architecture diagrams.\n\n"
            f"## Agreed Architecture\n{phase_a_context}\n\n"
            f"## Business Context\n{state.context_text}\n\n"
            f"## Data Sources\n{state.data_sources_text}\n"
        )

        if state.feedback_log:
            prompt += (
                "\n\n## Feedback to incorporate:\n"
                + "\n".join(f"- {fb}" for fb in state.feedback_log)
            )

        output = persona.invoke(prompt)

        return {
            "agent_outputs": {f"{persona.name}_build": output},
            "artifacts": [f"{persona.name}_build_output"],
        }

    return node


def _make_test_node(persona: BasePersona):
    """Create a Phase C node: persona tests and validates pipeline outputs."""

    def node(state: SwarmState) -> dict[str, Any]:
        logger.info("[Phase C] %s is testing pipeline outputs…", persona.name)

        build_context = "\n\n---\n\n".join(
            f"### {name}\n{output}"
            for name, output in state.agent_outputs.items()
            if "_build" in name
        )

        prompt = (
            "## Task: Test & Validate Pipeline\n\n"
            "You are in **Phase C — Test**.  Review and test the pipeline "
            "components built in Phase B.  From your expert perspective:\n"
            "1. Identify any bugs, logic errors, or edge cases.\n"
            "2. Validate data integrity assumptions.\n"
            "3. Check statistical methodology correctness.\n"
            "4. Generate test code if needed using `generate_code`.\n"
            "5. Rate overall quality: PASS / NEEDS_WORK / FAIL.\n\n"
            f"## Pipeline Components\n{build_context}\n\n"
            f"## Business Context\n{state.context_text}\n"
        )

        output = persona.invoke(prompt)
        satisfied = "PASS" in output.upper()

        return {
            "agent_outputs": {f"{persona.name}_test": output},
            "is_satisfied": {persona.name: satisfied},
        }

    return node


def _gate_1_node(state: SwarmState) -> dict[str, Any]:
    """Gate 1: human reviews architecture before build phase."""
    gate = HumanReviewGate("Gate 1 — Architecture Review")

    # Collect artefacts to display
    artefacts: dict[str, str] = {}
    artefact_types: dict[str, str] = {}

    for name, output in state.agent_outputs.items():
        if name.endswith("_review"):
            artefacts[f"Review: {name}"] = output
        else:
            artefacts[f"Analysis: {name}"] = output

    # Plain agent outputs for summary
    agent_summary = {
        name: output[:500]
        for name, output in state.agent_outputs.items()
        if not name.endswith("_review")
    }

    verdict, feedback = gate.run(
        artefacts=artefacts,
        agent_outputs=agent_summary,
        artefact_types=artefact_types,
    )

    if verdict == ReviewVerdict.APPROVE:
        return {"current_phase": Phase.BUILD}
    elif verdict == ReviewVerdict.REJECT:
        return {"current_phase": Phase.ABORTED}
    else:
        return {
            "current_phase": Phase.UNDERSTAND,
            "feedback_log": [feedback],
            "review_count": state.review_count + 1,
        }


def _gate_2_node(state: SwarmState) -> dict[str, Any]:
    """Gate 2: human reviews final pipeline before shipping."""
    gate = HumanReviewGate("Gate 2 — Final Review")

    artefacts: dict[str, str] = {}
    artefact_types: dict[str, str] = {}

    for name, output in state.agent_outputs.items():
        if "_build" in name or "_test" in name:
            artefacts[name] = output

    agent_summary = {
        name: output[:500]
        for name, output in state.agent_outputs.items()
    }

    verdict, feedback = gate.run(
        artefacts=artefacts,
        agent_outputs=agent_summary,
        artefact_types=artefact_types,
    )

    if verdict == ReviewVerdict.APPROVE:
        return {"current_phase": Phase.SHIP}
    elif verdict == ReviewVerdict.REJECT:
        return {"current_phase": Phase.ABORTED}
    else:
        return {
            "current_phase": Phase.BUILD,
            "feedback_log": [feedback],
            "review_count": state.review_count + 1,
        }


def _ship_node(state: SwarmState) -> dict[str, Any]:
    """Phase D: finalise outputs and log decisions."""
    logger.info("[Phase D] Shipping — finalising outputs…")
    return {"current_phase": Phase.DONE}


# ======================================================================
# Conditional edges
# ======================================================================

def _should_continue_review(state: SwarmState) -> str:
    """After reviews, decide whether to loop or proceed to gate."""
    if state.review_count >= state.max_review_loops:
        logger.warning("Max review loops (%d) reached — forcing gate.", state.max_review_loops)
        return "gate_1"

    if state.require_unanimous:
        all_happy = all(state.is_satisfied.values())
    else:
        happy_count = sum(1 for v in state.is_satisfied.values() if v)
        all_happy = happy_count >= len(state.is_satisfied) / 2

    return "gate_1" if all_happy else "review_loop"


def _should_continue_testing(state: SwarmState) -> str:
    """After testing, decide whether to loop or proceed to Gate 2."""
    if state.review_count >= state.max_review_loops:
        logger.warning("Max test loops (%d) reached — forcing gate.", state.max_review_loops)
        return "gate_2"

    all_pass = all(
        v for k, v in state.is_satisfied.items()
    )
    return "gate_2" if all_pass else "test_loop"


def _route_after_gate_1(state: SwarmState) -> str:
    """Route after Gate 1 based on the updated phase."""
    if state.current_phase == Phase.BUILD:
        return "build"
    elif state.current_phase == Phase.ABORTED:
        return "aborted"
    else:
        return "understand"


def _route_after_gate_2(state: SwarmState) -> str:
    """Route after Gate 2 based on the updated phase."""
    if state.current_phase == Phase.SHIP:
        return "ship"
    elif state.current_phase == Phase.ABORTED:
        return "aborted"
    else:
        return "build"


# ======================================================================
# Graph builder
# ======================================================================

def build_graph(personas: list[BasePersona], workflow_config: dict[str, Any]) -> StateGraph:
    """Construct the full LangGraph state machine.

    Parameters
    ----------
    personas : list[BasePersona]
        The agent personas to include in the pipeline.
    workflow_config : dict[str, Any]
        Workflow settings (max_review_loops, require_unanimous, etc.).

    Returns
    -------
    StateGraph
        A compiled LangGraph ready to be invoked.
    """
    graph = StateGraph(SwarmState)

    # ------------------------------------------------------------------
    # Phase A — Understand (one node per persona)
    # ------------------------------------------------------------------
    understand_nodes: list[str] = []
    for persona in personas:
        node_name = f"understand_{persona.config.persona_type}"
        graph.add_node(node_name, _make_understand_node(persona))
        understand_nodes.append(node_name)

    # Fan-out: START → all understand nodes
    # We chain them sequentially for simplicity (LangGraph processes in order)
    graph.set_entry_point(understand_nodes[0])
    for i in range(len(understand_nodes) - 1):
        graph.add_edge(understand_nodes[i], understand_nodes[i + 1])

    # ------------------------------------------------------------------
    # Review loop (each persona reviews others)
    # ------------------------------------------------------------------
    review_nodes: list[str] = []
    for persona in personas:
        node_name = f"review_{persona.config.persona_type}"
        graph.add_node(node_name, _make_review_node(persona, personas))
        review_nodes.append(node_name)

    # Last understand → first review
    graph.add_edge(understand_nodes[-1], review_nodes[0])
    for i in range(len(review_nodes) - 1):
        graph.add_edge(review_nodes[i], review_nodes[i + 1])

    # After last review: conditional — loop back or proceed to gate
    # Create a routing node
    def review_router(state: SwarmState) -> dict[str, Any]:
        """Passthrough node that increments review count."""
        return {"review_count": state.review_count + 1}

    graph.add_node("review_router", review_router)
    graph.add_edge(review_nodes[-1], "review_router")

    graph.add_conditional_edges(
        "review_router",
        _should_continue_review,
        {
            "gate_1": "gate_1",
            "review_loop": understand_nodes[0],
        },
    )

    # ------------------------------------------------------------------
    # Gate 1 — Architecture Review
    # ------------------------------------------------------------------
    graph.add_node("gate_1", _gate_1_node)
    graph.add_conditional_edges(
        "gate_1",
        _route_after_gate_1,
        {
            "build": "build_start",
            "aborted": "aborted",
            "understand": understand_nodes[0],
        },
    )

    # ------------------------------------------------------------------
    # Phase B — Build (one node per persona)
    # ------------------------------------------------------------------
    build_nodes: list[str] = []

    # Passthrough entry point for the build phase
    def build_start_fn(state: SwarmState) -> dict[str, Any]:
        logger.info("=== Entering Phase B — Build ===")
        return {"current_phase": Phase.BUILD}

    graph.add_node("build_start", build_start_fn)

    for persona in personas:
        node_name = f"build_{persona.config.persona_type}"
        graph.add_node(node_name, _make_build_node(persona))
        build_nodes.append(node_name)

    graph.add_edge("build_start", build_nodes[0])
    for i in range(len(build_nodes) - 1):
        graph.add_edge(build_nodes[i], build_nodes[i + 1])

    # ------------------------------------------------------------------
    # Phase C — Test loop
    # ------------------------------------------------------------------
    test_nodes: list[str] = []

    def test_start_fn(state: SwarmState) -> dict[str, Any]:
        logger.info("=== Entering Phase C — Test ===")
        return {"current_phase": Phase.TEST}

    graph.add_node("test_start", test_start_fn)
    graph.add_edge(build_nodes[-1], "test_start")

    for persona in personas:
        node_name = f"test_{persona.config.persona_type}"
        graph.add_node(node_name, _make_test_node(persona))
        test_nodes.append(node_name)

    graph.add_edge("test_start", test_nodes[0])
    for i in range(len(test_nodes) - 1):
        graph.add_edge(test_nodes[i], test_nodes[i + 1])

    # Test routing
    def test_router(state: SwarmState) -> dict[str, Any]:
        """Passthrough node after testing."""
        return {"review_count": state.review_count + 1}

    graph.add_node("test_router", test_router)
    graph.add_edge(test_nodes[-1], "test_router")

    graph.add_conditional_edges(
        "test_router",
        _should_continue_testing,
        {
            "gate_2": "gate_2",
            "test_loop": "build_start",
        },
    )

    # ------------------------------------------------------------------
    # Gate 2 — Final Review
    # ------------------------------------------------------------------
    graph.add_node("gate_2", _gate_2_node)
    graph.add_conditional_edges(
        "gate_2",
        _route_after_gate_2,
        {
            "ship": "ship",
            "aborted": "aborted",
            "build": "build_start",
        },
    )

    # ------------------------------------------------------------------
    # Phase D — Ship
    # ------------------------------------------------------------------
    graph.add_node("ship", _ship_node)
    graph.add_edge("ship", END)

    # ------------------------------------------------------------------
    # Aborted terminal node
    # ------------------------------------------------------------------
    def aborted_fn(state: SwarmState) -> dict[str, Any]:
        logger.warning("Pipeline aborted by reviewer.")
        return {"current_phase": Phase.ABORTED}

    graph.add_node("aborted", aborted_fn)
    graph.add_edge("aborted", END)

    # ------------------------------------------------------------------
    # Compile and return
    # ------------------------------------------------------------------
    compiled = graph.compile()
    logger.info(
        "Built swarm graph with %d understand, %d review, %d build, %d test nodes",
        len(understand_nodes),
        len(review_nodes),
        len(build_nodes),
        len(test_nodes),
    )
    return compiled
