"""Best-practice search tool.

Provides a LangChain tool that agents can call to look up industry best
practices for a given topic.  Currently implemented as an LLM stub — the
user can later wire this up to a real web search or RAG system.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# The stub LLM is created lazily to avoid import-time API key requirements.
_llm = None


def _get_llm():
    """Lazy-init the stub LLM for best-practice queries."""
    global _llm
    if _llm is None:
        from langchain_openai import ChatOpenAI

        _llm = ChatOpenAI(
            model="gpt-5.4-mini",
            temperature=0.3,
            max_tokens=2048,
        )
    return _llm


@tool
def best_practice_search(query: str) -> str:
    """Search for industry best practices on a given topic.

    Use this tool when you need to know the recommended approach, common
    pitfalls, or industry benchmarks for a data science or business problem.

    Args:
        query: A natural-language question about best practices, e.g.
               "What is the best practice for handling class imbalance in
               churn prediction?" or "Industry benchmarks for FMCG new
               product launch success rates".

    Returns:
        A structured best-practice summary with recommendations, common
        pitfalls, and references where available.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    logger.info("best_practice_search: %s", query)

    system = (
        "You are a senior industry analyst and data science methodologist. "
        "When asked about best practices, provide:\n"
        "1. **Recommended Approach** — the consensus best practice.\n"
        "2. **Common Pitfalls** — mistakes to avoid.\n"
        "3. **Benchmarks** — typical numbers or thresholds if applicable.\n"
        "4. **References** — mention seminal papers, frameworks, or tools "
        "(no URLs needed, just names).\n\n"
        "Be concise but actionable.  Stick to what is well-established."
    )

    llm = _get_llm()
    response = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=query),
        ]
    )
    return response.content  # type: ignore[return-value]
