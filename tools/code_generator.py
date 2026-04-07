"""Code generation tool.

Generates Python files based on a specification and saves them to the
``pipeline/`` directory.  Agents use this tool to produce analysis scripts,
data processing modules, and test files.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_llm = None

# Resolve the pipeline directory relative to this project
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PIPELINE_DIR = _PROJECT_ROOT / "pipeline"


def _get_llm():
    """Lazy-init the code generation LLM."""
    global _llm
    if _llm is None:
        from langchain_openai import ChatOpenAI

        _llm = ChatOpenAI(
            model="gpt-5.4",
            temperature=0.2,
            max_tokens=8192,
        )
    return _llm


@tool
def generate_code(specification: str, filename: str) -> str:
    """Generate a Python file from a specification and save it to the pipeline directory.

    Use this tool to create data processing scripts, analysis modules, test
    suites, or any other Python file required by the project.

    Args:
        specification: A detailed description of what the code should do,
                       including inputs, outputs, key algorithms, and any
                       constraints (e.g. "must use pandas, handle NaNs").
        filename: The target filename (e.g. ``01_data_cleaning.py``).  The
                  file will be saved under ``pipeline/{filename}``.

    Returns:
        The full path to the generated file, or an error message if
        generation failed.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    logger.info("generate_code: creating %s", filename)

    system = (
        "You are an expert Python developer specialising in data science pipelines. "
        "Generate a single, complete Python file based on the specification below.\n\n"
        "Requirements:\n"
        "- Include a module-level docstring explaining purpose and usage.\n"
        "- Use type hints on all function signatures.\n"
        "- Add docstrings (Google style) to every function.\n"
        "- Include proper error handling (try/except with meaningful messages).\n"
        "- Add a ``if __name__ == '__main__':`` block for standalone execution.\n"
        "- Use logging instead of print statements.\n"
        "- Import only standard-library packages plus: pandas, numpy, scikit-learn, "
        "scipy, matplotlib, seaborn (as needed).\n"
        "- Set random seeds for reproducibility where applicable.\n\n"
        "Return ONLY the Python code — no markdown fences, no explanatory text."
    )

    llm = _get_llm()
    response = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=specification),
        ]
    )

    code = response.content  # type: ignore[union-attr]

    # Strip any accidental markdown fences
    if isinstance(code, str):
        code = code.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            # Remove first and last lines (``` markers)
            lines = [l for l in lines if not l.strip().startswith("```")]
            code = "\n".join(lines)

    # Ensure pipeline directory exists
    _PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

    file_path = _PIPELINE_DIR / filename
    file_path.write_text(code, encoding="utf-8")

    logger.info("generate_code: saved %s (%d bytes)", file_path, len(code))
    return str(file_path)
