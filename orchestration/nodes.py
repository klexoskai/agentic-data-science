"""
AIQ-style graph nodes, mapped to your existing agentic-data-science stack.

Node layout mirrors AI-Q's canonical flow:

    orchestration_node
        ├── meta_node          (QueryIntent.META)
        └── shallow_node       (ResearchDepth.SHALLOW)
        └── deep_planner_node  ──► deep_researcher_node (loop) ──► synthesiser_node
                                        (ResearchDepth.DEEP)

All nodes write into ResearchState and are wired together in graph.py.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from orchestration.state import QueryIntent, ResearchDepth, ResearchState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llm(model: str = "gpt-4o-mini", temperature: float = 0.0) -> ChatOpenAI:
    """Thin factory so you can swap the model in config.yaml later."""
    return ChatOpenAI(model=model, temperature=temperature)


def _record_tool_call(state: ResearchState, tool: str, input_: Any, output: Any) -> dict:
    """Return a state patch that appends a tool call trace entry."""
    return {
        "tool_calls": [
            {"tool": tool, "input": str(input_)[:500], "output": str(output)[:500]}
        ]
    }


# ---------------------------------------------------------------------------
# 1. Orchestration node  (AI-Q: "orchestration node")
#    - Classifies intent
#    - Decides depth (shallow vs. deep)
# ---------------------------------------------------------------------------

_ORCHESTRATION_SYSTEM = """\
You are a research orchestrator. Given a user query, output a JSON object with:
  - "intent": "meta" if the query is about your capabilities, asking for a
    rephrasing, or is conversational. Otherwise "research".
  - "depth": "shallow" for straightforward questions answerable with 1-2 data
    lookups; "deep" for complex analytical questions that require multi-step
    planning, synthesis across many sources, or long-form reports.
  - "rationale": one sentence explaining your decision.

Return ONLY valid JSON. No markdown fences.

Examples:
  Q: "What is our YTD revenue in SEA?"
  → {"intent": "research", "depth": "shallow", "rationale": "Single metric lookup."}

  Q: "Give me a full market sizing and competitive analysis for Q3 FMCG in SEA."
  → {"intent": "research", "depth": "deep", "rationale": "Multi-source synthesis needed."}

  Q: "What can you help me with?"
  → {"intent": "meta", "depth": "shallow", "rationale": "Capability question."}
"""


def orchestration_node(state: ResearchState) -> dict:
    """Classify intent and decide research depth."""
    logger.info("[Orchestrator] Classifying query: %r", state.query[:80])

    response = _llm(temperature=0.0).invoke([
        SystemMessage(content=_ORCHESTRATION_SYSTEM),
        HumanMessage(content=state.query),
    ])

    try:
        parsed = json.loads(response.content)
        intent = QueryIntent(parsed.get("intent", "research"))
        depth = ResearchDepth(parsed.get("depth", "shallow"))
    except (json.JSONDecodeError, ValueError):
        logger.warning("[Orchestrator] Parse failed — defaulting to research/shallow.")
        intent = QueryIntent.RESEARCH
        depth = ResearchDepth.SHALLOW

    logger.info("[Orchestrator] intent=%s  depth=%s", intent, depth)
    return {"intent": intent, "depth": depth}


# ---------------------------------------------------------------------------
# 2. Meta node  (AI-Q: "clarifier / meta agent")
#    Handles capability questions, rephrasing, conversational turns.
# ---------------------------------------------------------------------------

_META_SYSTEM = """\
You are a helpful assistant. The user has asked a meta question (about capabilities,
rephrasing, or general conversation). Answer concisely and helpfully.
You have access to a multi-agent data science pipeline that can:
- Analyse FMCG / CPG sales data
- Generate forecasts and trend summaries
- Run multi-agent peer review on analytical conclusions
- Produce reports with supporting citations
"""


def meta_node(state: ResearchState) -> dict:
    logger.info("[Meta] Handling meta query.")
    response = _llm(temperature=0.3).invoke([
        SystemMessage(content=_META_SYSTEM),
        HumanMessage(content=state.query),
    ])
    return {"answer": response.content, "intent": QueryIntent.META}


# ---------------------------------------------------------------------------
# 3. Shallow research node  (AI-Q: "shallow research agent")
#    Bounded: up to ~5 tool calls, returns answer + citations quickly.
# ---------------------------------------------------------------------------

_SHALLOW_SYSTEM = """\
You are a research analyst. Answer the user's question using the retrieved
context below. Be concise. Cite every factual claim with a [source_N] marker
referencing the chunk index. End with a "## Sources" section listing each
cited chunk.

Context:
{context}
"""


def shallow_node(state: ResearchState) -> dict:
    """Fast, citation-backed answer from retrieved context."""
    logger.info("[Shallow] Generating answer from %d chunks.", len(state.retrieved_chunks))

    context_text = _format_chunks(state.retrieved_chunks)
    prompt = _SHALLOW_SYSTEM.format(context=context_text or "(no context retrieved yet)")

    response = _llm(temperature=0.2).invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=state.query),
    ])

    citations = _extract_citations(state.retrieved_chunks)
    return {"answer": response.content, "citations": citations}


# ---------------------------------------------------------------------------
# 4. Deep planner node  (AI-Q: "deep research agent — planning phase")
#    Decomposes the query into a list of subtasks / sub-questions.
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = """\
You are a research planner. Break the user's complex question into 3-5 concrete
sub-questions or subtasks that together fully answer it. Each subtask should be
independently researchable.

Return a JSON array of strings. No markdown fences.

Example:
["What is the YTD revenue trend by SKU?",
 "Which markets drove the most growth?",
 "What are the top risk factors for Q4?"]
"""


def deep_planner_node(state: ResearchState) -> dict:
    """Decompose query into a sub-task plan."""
    logger.info("[DeepPlanner] Decomposing query into subtasks.")

    response = _llm(temperature=0.1).invoke([
        SystemMessage(content=_PLANNER_SYSTEM),
        HumanMessage(content=state.query),
    ])

    try:
        plan = json.loads(response.content)
    except (json.JSONDecodeError, TypeError):
        logger.warning("[DeepPlanner] Parse failed — using single-task plan.")
        plan = [state.query]

    logger.info("[DeepPlanner] Plan: %s", plan)
    return {"plan": plan, "iteration": 0}


# ---------------------------------------------------------------------------
# 5. Deep researcher node  (AI-Q: "deep research agent — research loop")
#    Iterates over the plan, calls tools, accumulates findings.
#    Mirrors your existing _make_build_node / _make_understand_node pattern.
# ---------------------------------------------------------------------------

_RESEARCHER_SYSTEM = """\
You are a senior data scientist researching the following subtask:

Subtask: {subtask}

Use the retrieved context below to produce findings. Be specific, reference
data where available. Flag gaps where context is insufficient.

Context:
{context}
"""


def deep_researcher_node(state: ResearchState) -> dict:
    """Research the current subtask (one iteration of the deep loop)."""
    if not state.plan:
        return {"iteration": state.max_iterations}  # force exit

    subtask_idx = state.iteration % len(state.plan)
    subtask = state.plan[subtask_idx]
    logger.info("[DeepResearcher] Iteration %d — subtask: %r", state.iteration, subtask[:60])

    context_text = _format_chunks(state.retrieved_chunks)
    prompt = _RESEARCHER_SYSTEM.format(subtask=subtask, context=context_text or "(none yet)")

    response = _llm(temperature=0.3).invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=state.query),
    ])

    patch = {
        "agent_outputs": {f"deep_iter_{state.iteration}": response.content},
        "iteration": state.iteration + 1,
    }
    patch.update(_record_tool_call(state, "deep_researcher", subtask, response.content[:200]))
    return patch


# ---------------------------------------------------------------------------
# 6. Synthesiser node  (AI-Q: "deep research agent — synthesis phase")
#    Combines all deep research iterations into a final report with citations.
# ---------------------------------------------------------------------------

_SYNTHESISER_SYSTEM = """\
You are a senior analyst writing a final research report. Synthesise the
findings below into a coherent, structured answer to the original question.

Format:
## Executive Summary
(2-3 sentences)

## Key Findings
(bullet points with [source_N] citations)

## Detailed Analysis
(narrative with citations)

## Caveats & Data Gaps
(brief)

## Sources
(list every cited chunk)

Original question: {query}

Findings from research iterations:
{findings}

Retrieved context:
{context}
"""


def synthesiser_node(state: ResearchState) -> dict:
    """Synthesise all deep research iterations into a final report."""
    logger.info("[Synthesiser] Synthesising %d research iterations.", state.iteration)

    findings = "\n\n---\n\n".join(
        f"### Iteration {k.replace('deep_iter_', '')}\n{v}"
        for k, v in sorted(state.agent_outputs.items())
        if k.startswith("deep_iter_")
    )
    context_text = _format_chunks(state.retrieved_chunks)

    prompt = _SYNTHESISER_SYSTEM.format(
        query=state.query,
        findings=findings or "(no findings yet)",
        context=context_text or "(no context)",
    )

    response = _llm(temperature=0.2).invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Produce the final report now."),
    ])

    citations = _extract_citations(state.retrieved_chunks)
    return {"answer": response.content, "citations": citations}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _format_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    parts = []
    for i, chunk in enumerate(chunks):
        source = chunk.get("source", "unknown")
        text = chunk.get("text", "")
        parts.append(f"[source_{i}] ({source})\n{text}")
    return "\n\n".join(parts)


def _extract_citations(chunks: list[dict]) -> list[str]:
    return [
        f"[source_{i}] {chunk.get('source', 'unknown')}"
        for i, chunk in enumerate(chunks)
    ]
