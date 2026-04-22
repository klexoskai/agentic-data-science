"""
tools/web_search.py — Tavily web search tool for agent use.

Provides two interfaces:
  1. `web_search_tool`  — a @tool-decorated function for LangChain tool binding.
     Pass this to llm.bind_tools([web_search_tool]) so the LLM can decide
     when to invoke it mid-reasoning.

  2. `web_search(query)`  — a plain Python function for direct calls from
     node functions (e.g. strategy_nodes.py enrichment step).

Requires:
    TAVILY_API_KEY in your .env
    pip install tavily-python

Free tier: 1,000 searches/month — https://tavily.com
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plain Python interface (direct call from node functions)
# ---------------------------------------------------------------------------

def web_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "advanced",
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> list[dict[str, str]]:
    """
    Run a Tavily web search and return structured results.

    Parameters
    ----------
    query : str
        The search query.
    max_results : int
        Max number of results to return (default 5).
    search_depth : str
        "basic" (faster) or "advanced" (more thorough, uses 2 API credits).
    include_domains : list[str] | None
        Restrict results to these domains, e.g. ["pubmed.ncbi.nlm.nih.gov"].
    exclude_domains : list[str] | None
        Exclude these domains from results.

    Returns
    -------
    list[dict]  — each dict has: title, url, content, score
    """
    try:
        from tavily import TavilyClient
    except ImportError:
        raise ImportError("Run: pip install tavily-python")

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise EnvironmentError("TAVILY_API_KEY not set in environment / .env")

    client = TavilyClient(api_key=api_key)

    kwargs: dict[str, Any] = {
        "query": query,
        "max_results": max_results,
        "search_depth": search_depth,
    }
    if include_domains:
        kwargs["include_domains"] = include_domains
    if exclude_domains:
        kwargs["exclude_domains"] = exclude_domains

    logger.info("[WebSearch] Query: %r (depth=%s)", query[:80], search_depth)

    try:
        response = client.search(**kwargs)
    except Exception as exc:
        logger.warning("[WebSearch] Failed: %s", exc)
        return []

    results = []
    for r in response.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "score": str(r.get("score", "")),
        })

    logger.info("[WebSearch] Returned %d results.", len(results))
    return results


def web_search_to_chunks(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """
    Convenience wrapper: run web_search and return results in the same
    {"source": str, "text": str} chunk format used by the ChromaDB retriever.
    Allows seamlessly mixing web results with stored context.
    """
    raw = web_search(query, max_results=max_results)
    return [
        {
            "source": r["url"],
            "text": f"{r['title']}\n{r['content']}",
            "metadata": {"url": r["url"], "score": r["score"]},
            "distance": 1.0 - float(r["score"]) if r["score"] else 0.5,
        }
        for r in raw
    ]


# ---------------------------------------------------------------------------
# LangChain @tool interface (for llm.bind_tools())
# ---------------------------------------------------------------------------

@tool
def web_search_tool(query: str) -> str:
    """
    Search the web for current information relevant to the query.
    Use this when you need external market data, industry benchmarks,
    recent news, academic references, or any information not in the
    internal knowledge base.

    Returns a formatted string of search results with titles, URLs,
    and content summaries.
    """
    results = web_search(query, max_results=5)
    if not results:
        return "No results found."

    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(
            f"[Result {i}] {r['title']}\n"
            f"URL: {r['url']}\n"
            f"{r['content']}\n"
        )
    return "\n---\n".join(formatted)


@tool
def chroma_search_tool(query: str) -> str:
    """
    Search the internal ChromaDB knowledge base for relevant context,
    including internal data files (P&L, launch tracker, TM1, IQVIA, etc.)
    and project documentation (context.md, data_sources.md, ADRs).

    Use this when you need internal data, historical actuals, or
    project-specific information.
    """
    try:
        from store.retriever import retrieve
        chunks = retrieve(query, collection="all", n_results=5)
    except Exception as exc:
        return f"ChromaDB unavailable: {exc}"

    if not chunks:
        return "No relevant internal context found."

    formatted = []
    for i, c in enumerate(chunks, 1):
        formatted.append(
            f"[Internal {i}] Source: {c['source']}\n{c['text'][:600]}"
        )
    return "\n---\n".join(formatted)


# All tools exposed for easy import
ALL_TOOLS = [web_search_tool, chroma_search_tool]
