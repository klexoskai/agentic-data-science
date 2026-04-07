"""Diagram generation tool.

Generates Mermaid ``.mmd`` diagram files from architecture or data-flow
descriptions.  Useful for producing visual documentation that can be rendered
in GitHub, VS Code, or any Mermaid-compatible viewer.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_llm = None

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OUTPUTS_DIR = _PROJECT_ROOT / "outputs"


def _get_llm():
    """Lazy-init the diagram generation LLM."""
    global _llm
    if _llm is None:
        from langchain_openai import ChatOpenAI

        _llm = ChatOpenAI(
            model="gpt-5.4-mini",
            temperature=0.2,
            max_tokens=4096,
        )
    return _llm


@tool
def generate_diagram(description: str, filename: str, diagram_type: str = "flowchart") -> str:
    """Generate a Mermaid diagram and save it as an .mmd file.

    Use this tool to create architecture diagrams, data flow diagrams, or
    module dependency graphs as part of the pipeline documentation.

    Args:
        description: A natural-language description of what the diagram should
                     show, e.g. "Data flow from raw CSV sources through
                     cleaning, feature engineering, model training, to
                     dashboard output."
        filename: Target filename without extension (e.g. ``architecture``).
                  Will be saved as ``outputs/{filename}.mmd``.
        diagram_type: One of ``flowchart``, ``sequenceDiagram``, ``classDiagram``,
                      ``erDiagram``, ``gantt``.  Defaults to ``flowchart``.

    Returns:
        The full path to the generated ``.mmd`` file.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    logger.info("generate_diagram: creating %s (%s)", filename, diagram_type)

    system = (
        "You are an expert at creating Mermaid diagrams.  Given a description, "
        f"produce a valid Mermaid **{diagram_type}** diagram.\n\n"
        "Rules:\n"
        "- Output ONLY the Mermaid code — no markdown fences, no explanation.\n"
        "- Use clear, readable node labels.\n"
        "- Keep it clean — avoid overly complex layouts.\n"
        "- Use subgraphs to group related components where appropriate.\n"
        "- Use meaningful edge labels to describe data flow or relationships."
    )

    llm = _get_llm()
    response = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=description),
        ]
    )

    mermaid_code = response.content  # type: ignore[union-attr]

    # Strip accidental markdown fences
    if isinstance(mermaid_code, str):
        mermaid_code = mermaid_code.strip()
        if mermaid_code.startswith("```"):
            lines = mermaid_code.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            mermaid_code = "\n".join(lines)

    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = _OUTPUTS_DIR / f"{filename}.mmd"
    file_path.write_text(mermaid_code, encoding="utf-8")

    logger.info("generate_diagram: saved %s (%d bytes)", file_path, len(mermaid_code))
    return str(file_path)
