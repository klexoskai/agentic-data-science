"""
orchestration/strategy_nodes.py — Node functions for the Strategy Council graph.

Flow:
    enrichment_node         — fetch internal (ChromaDB) + web context for the problem
        ↓
    council_node x3         — Data Scientist, Sales Director, QA Engineer each
                              independently propose a data science pipeline strategy
        ↓
    debate_node             — agents read each other's proposals and surface conflicts/gaps
        ↓
    critic_node (x2 rounds) — a dedicated Critic LLM fact-checks and stress-tests
                              the debated strategy; flags weak assumptions
        ↓
    synthesiser_node        — produces the final comprehensive strategy document

Each node is a plain function: (StrategyState) → dict patch.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from orchestration.strategy_state import CouncilMember, StrategyState
from tools.web_search import ALL_TOOLS, web_search_to_chunks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _llm(model: str = "gpt-4o-mini", temperature: float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=temperature)


def _llm_with_tools(model: str = "gpt-4o-mini", temperature: float = 0.3) -> ChatOpenAI:
    """LLM with web search + ChromaDB tools bound for autonomous tool use."""
    return ChatOpenAI(model=model, temperature=temperature).bind_tools(ALL_TOOLS)


# ---------------------------------------------------------------------------
# 1. Enrichment node
#    Gathers internal (ChromaDB) + web context before agents begin.
#    Enriched chunks are written to state.retrieved_chunks.
# ---------------------------------------------------------------------------

_ENRICHMENT_WEB_QUERIES = [
    "FMCG OTC healthcare data science pipeline best practices",
    "sales forecasting methodology FMCG Southeast Asia",
    "launch similarity analysis pharmaceutical consumer health",
]


def enrichment_node(state: StrategyState) -> dict:
    """Pre-fetch internal + web context to ground all council agents."""
    logger.info("[Enrichment] Fetching internal + web context…")

    chunks: list[dict] = []

    # 1. Internal ChromaDB retrieval
    try:
        from store.retriever import retrieve_for_query
        internal = retrieve_for_query(
            query=state.problem_statement,
            context_text="\n".join(state.dataset_descriptions),
            n_results=8,
        )
        chunks.extend(internal)
        logger.info("[Enrichment] %d internal chunks retrieved.", len(internal))
    except Exception as exc:
        logger.warning("[Enrichment] ChromaDB unavailable: %s", exc)

    # 2. Web search — problem-specific query
    try:
        web = web_search_to_chunks(state.problem_statement, max_results=4)
        chunks.extend(web)
        logger.info("[Enrichment] %d web chunks retrieved.", len(web))
    except Exception as exc:
        logger.warning("[Enrichment] Web search unavailable: %s", exc)

    # 3. Domain-specific web queries
    for q in _ENRICHMENT_WEB_QUERIES:
        try:
            web = web_search_to_chunks(q, max_results=2)
            chunks.extend(web)
        except Exception:
            pass

    return {"retrieved_chunks": chunks}


# ---------------------------------------------------------------------------
# 2. Council nodes — one per agent persona
#    Each agent independently proposes a full data science pipeline strategy.
# ---------------------------------------------------------------------------

_COUNCIL_SYSTEM = """\
You are a {role} with deep expertise in {domain}.

You are part of a strategy council tasked with designing a comprehensive
data science pipeline to solve the business problem below.

Your job: propose a COMPLETE strategy from your expert perspective, covering:
1. **Problem Interpretation** — how you frame the problem from your lens
2. **Recommended Pipeline Architecture** — end-to-end steps with rationale
3. **Dataset Usage** — which datasets to use, how to join/transform them, known issues
4. **Modelling / Analytical Approach** — specific methods, algorithms, or frameworks
5. **Evaluation Criteria** — how you'd measure success
6. **Caveats & Limitations** — honest assessment of data gaps, assumptions, risks
7. **Open Questions** — what you'd need answered before building

Use the retrieved context below to ground your proposal in real data and benchmarks.
Be specific. Do not hedge excessively — take a position.

Retrieved context:
{context}
"""


def _format_chunks(chunks: list[dict], max_chars: int = 4000) -> str:
    parts = []
    total = 0
    for i, c in enumerate(chunks):
        snippet = f"[{c.get('source', '?')}]\n{c.get('text', '')[:600]}"
        if total + len(snippet) > max_chars:
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n\n---\n\n".join(parts) or "(no context retrieved)"


_COUNCIL_MEMBERS = [
    CouncilMember(
        key="data_scientist",
        role="Senior Data Scientist",
        domain="statistical modelling, forecasting, and ML pipelines for FMCG / healthcare",
    ),
    CouncilMember(
        key="sales_director",
        role="Sales Director",
        domain="FMCG commercial strategy, SKU portfolio management, and SEA market dynamics",
    ),
    CouncilMember(
        key="qa_engineer",
        role="Data Quality & QA Engineer",
        domain="data pipeline validation, schema integrity, and statistical assumption testing",
    ),
]


def _make_council_node(member: CouncilMember, model: str = "gpt-4o-mini"):
    """Factory: returns a node function for a specific council member."""

    def node(state: StrategyState) -> dict:
        logger.info("[Council] %s (%s) drafting strategy…", member.role, member.key)

        context_text = _format_chunks(state.retrieved_chunks)
        datasets_text = "\n".join(
            f"- {d}" for d in state.dataset_descriptions
        ) or "(no datasets specified)"

        system = _COUNCIL_SYSTEM.format(
            role=member.role,
            domain=member.domain,
            context=context_text,
        )
        user = (
            f"## Business Problem\n{state.problem_statement}\n\n"
            f"## Available Datasets\n{datasets_text}\n\n"
            "Draft your strategy proposal now."
        )

        response = _llm(model=model, temperature=0.4).invoke([
            SystemMessage(content=system),
            HumanMessage(content=user),
        ])

        return {
            "proposals": {member.key: response.content},
        }

    node.__name__ = f"council_{member.key}"
    return node


# Build the three council nodes (accessed by graph.py)
council_nodes = [_make_council_node(m) for m in _COUNCIL_MEMBERS]
COUNCIL_MEMBER_KEYS = [m.key for m in _COUNCIL_MEMBERS]


# ---------------------------------------------------------------------------
# 3. Debate node
#    Each agent reads all other proposals and surfaces conflicts, gaps,
#    and synthesis points. Output feeds the critic.
# ---------------------------------------------------------------------------

_DEBATE_SYSTEM = """\
You are a senior research moderator. Three domain experts have each
independently proposed a data science pipeline strategy for the business
problem below.

Your task:
1. **Identify agreements** — where all three proposals align (high confidence areas)
2. **Surface conflicts** — where proposals disagree or contradict each other
3. **Flag gaps** — important considerations missing from ALL proposals
4. **Preliminary synthesis** — a rough merged strategy that resolves conflicts
   by choosing the best-supported position, with explicit rationale

Be rigorous. Do not paper over real disagreements. If proposals conflict,
state which position you believe is stronger and why.

Proposals:
{proposals}

Business Problem:
{problem}

Available Datasets:
{datasets}
"""


def debate_node(state: StrategyState) -> dict:
    """Moderate the three proposals into a preliminary synthesis."""
    logger.info("[Debate] Moderating %d proposals…", len(state.proposals))

    proposals_text = "\n\n===\n\n".join(
        f"### {key.replace('_', ' ').title()}\n{text}"
        for key, text in sorted(state.proposals.items())
    )
    datasets_text = "\n".join(f"- {d}" for d in state.dataset_descriptions)

    prompt = _DEBATE_SYSTEM.format(
        proposals=proposals_text,
        problem=state.problem_statement,
        datasets=datasets_text,
    )

    response = _llm(temperature=0.2).invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Produce the moderated synthesis now."),
    ])

    return {"debate_summary": response.content}


# ---------------------------------------------------------------------------
# 4. Critic node (runs twice)
#    A dedicated critic stress-tests the debated strategy.
#    Flags weak assumptions, data quality risks, and logical gaps.
#    Second round receives the first critique and checks if issues were addressed.
# ---------------------------------------------------------------------------

_CRITIC_SYSTEM_ROUND_1 = """\
You are a rigorous senior analyst acting as a devil's advocate.
Your job is to STRESS-TEST the proposed data science strategy below.

For each section of the strategy, ask:
- Is this assumption realistic given the data described?
- Is this methodology appropriate for the data volume/quality available?
- Could this step fail silently, producing wrong results without obvious errors?
- What would need to be true for this to work, and is that likely?
- Are there simpler approaches that would be more robust?

Structure your critique as:
## Critical Issues (must fix before building)
## Moderate Concerns (should address)
## Minor Notes (nice to have)
## Verdict: APPROVED / NEEDS_REVISION / REJECTED
   (with a one-sentence rationale)

Be specific. Quote the strategy text you are critiquing.

Strategy to critique:
{strategy}

Business Problem:
{problem}

Available Datasets:
{datasets}

Retrieved Context (data quality / benchmarks):
{context}
"""

_CRITIC_SYSTEM_ROUND_2 = """\
You are a rigorous senior analyst performing a SECOND PASS critique.

The strategy was revised after the first critique. Check:
1. Were the Critical Issues from round 1 genuinely addressed, or just acknowledged?
2. Were the Moderate Concerns resolved or explicitly deferred with good reason?
3. Are there any NEW issues introduced by the revisions?
4. Final verdict: is this strategy robust enough to hand to an engineering team?

First critique:
{prior_critique}

Revised strategy:
{strategy}

Business Problem:
{problem}
"""


def make_critic_node(round_number: int, model: str = "gpt-4o-mini"):
    """Factory: returns a critic node for the given round (1 or 2)."""

    def node(state: StrategyState) -> dict:
        logger.info("[Critic] Round %d critique…", round_number)

        context_text = _format_chunks(state.retrieved_chunks, max_chars=2000)
        datasets_text = "\n".join(f"- {d}" for d in state.dataset_descriptions)

        # Strategy to critique = latest synthesised output, or debate summary
        strategy = state.final_strategy or state.debate_summary

        if round_number == 1:
            system = _CRITIC_SYSTEM_ROUND_1.format(
                strategy=strategy,
                problem=state.problem_statement,
                datasets=datasets_text,
                context=context_text,
            )
        else:
            # Round 2: critique the first critique's resolution
            prior = state.critiques[0] if state.critiques else "(no prior critique)"
            system = _CRITIC_SYSTEM_ROUND_2.format(
                prior_critique=prior,
                strategy=strategy,
                problem=state.problem_statement,
            )

        response = _llm(model=model, temperature=0.1).invoke([
            SystemMessage(content=system),
            HumanMessage(content=f"Produce round {round_number} critique now."),
        ])

        return {"critiques": [response.content]}

    node.__name__ = f"critic_round_{round_number}"
    return node


critic_node_1 = make_critic_node(1)
critic_node_2 = make_critic_node(2)


# ---------------------------------------------------------------------------
# 5. Synthesiser node
#    Produces the final polished strategy document incorporating
#    all proposals, the debate, and both rounds of critique.
# ---------------------------------------------------------------------------

_SYNTHESISER_SYSTEM = """\
You are a principal data scientist writing the FINAL strategy document.
Synthesise all inputs below into a single, comprehensive, publication-ready
data science pipeline strategy.

The document must follow this structure:

# Data Science Pipeline Strategy
## Executive Summary (3-5 sentences — what we're building and why)

## Problem Statement & Scope
## Dataset Inventory & Data Quality Assessment
## Recommended Pipeline Architecture
   ### Phase 1: Data Preparation & Integration
   ### Phase 2: Exploratory Analysis & Feature Engineering
   ### Phase 3: Modelling Approach
   ### Phase 4: Evaluation & Validation
   ### Phase 5: Deployment & Monitoring

## Key Assumptions
## Caveats & Limitations (be explicit — do not hide risks)
## Open Questions & Prerequisites
## Recommended Next Steps

Rules:
- Every factual claim must reference its source (internal dataset or [Web: URL]).
- Where proposals conflicted, state which position was chosen and why.
- Where critiques raised unresolved issues, flag them explicitly under Caveats.
- Be specific about data transformations, join keys, and model choices.
- Do not be vague to avoid controversy. Take positions.

Inputs:

### Business Problem
{problem}

### Datasets
{datasets}

### Agent Proposals
{proposals}

### Debate Synthesis
{debate}

### Critique Round 1
{critique_1}

### Critique Round 2
{critique_2}

### Retrieved Context
{context}
"""


def synthesiser_node(state: StrategyState) -> dict:
    """Produce the final polished strategy document."""
    logger.info("[Synthesiser] Producing final strategy document…")

    proposals_text = "\n\n---\n\n".join(
        f"### {k.replace('_', ' ').title()}\n{v}"
        for k, v in sorted(state.proposals.items())
    )
    datasets_text = "\n".join(f"- {d}" for d in state.dataset_descriptions)
    context_text = _format_chunks(state.retrieved_chunks, max_chars=3000)
    critiques = state.critiques
    critique_1 = critiques[0] if len(critiques) > 0 else "(none)"
    critique_2 = critiques[1] if len(critiques) > 1 else "(none)"

    system = _SYNTHESISER_SYSTEM.format(
        problem=state.problem_statement,
        datasets=datasets_text,
        proposals=proposals_text,
        debate=state.debate_summary or "(none)",
        critique_1=critique_1,
        critique_2=critique_2,
        context=context_text,
    )

    response = _llm(temperature=0.2).invoke([
        SystemMessage(content=system),
        HumanMessage(content="Produce the final strategy document now."),
    ])

    return {"final_strategy": response.content}
