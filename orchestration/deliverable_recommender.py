"""
orchestration/deliverable_recommender.py — Deliverable Recommender Agent.

Reads the final strategy document and recommends the most appropriate
end product (dashboard, pipeline, API, report, etc.), then generates
a concrete build plan using the actual raw files in data/.

The recommender produces a structured DeliverableSpec that is:
  1. Displayed to the user for approval at the strategy gate
  2. Passed to the existing pipeline's build phase if approved

Deliverable types:
  - dashboard   : Plotly Dash interactive app
  - pipeline    : Standalone Python ETL/ML pipeline script
  - report      : Automated markdown/HTML report generator
  - api         : FastAPI REST endpoint serving model predictions
  - notebook    : Jupyter notebook with narrative + code
  - hybrid      : Combination (e.g. pipeline + dashboard)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DeliverableSpec — the structured output of the recommender
# ---------------------------------------------------------------------------

@dataclass
class DeliverableSpec:
    """Structured recommendation for the end product."""

    deliverable_type: str           # dashboard | pipeline | report | api | notebook | hybrid
    title: str                      # short name, e.g. "SEA Launch Forecast Dashboard"
    rationale: str                  # why this deliverable type fits the problem
    target_user: str                # who will use it (e.g. "Commercial team, monthly")
    tech_stack: list[str]           # e.g. ["Python", "Pandas", "Plotly Dash"]
    input_datasets: list[str]       # dataset filenames from data/ that will be used
    key_features: list[str]         # concrete features / outputs to build
    phased_build_plan: list[dict]   # [{phase: str, tasks: [str], files: [str]}]
    estimated_complexity: str       # low | medium | high
    caveats: list[str]              # limitations / prerequisites

    def to_markdown(self) -> str:
        """Render spec as a readable markdown summary for the approval gate."""
        lines = [
            f"# Recommended Deliverable: {self.title}",
            f"\n**Type:** `{self.deliverable_type}`  |  "
            f"**Complexity:** `{self.estimated_complexity}`  |  "
            f"**Target user:** {self.target_user}",
            f"\n## Rationale\n{self.rationale}",
            f"\n## Tech Stack\n" + "\n".join(f"- {t}" for t in self.tech_stack),
            f"\n## Input Datasets\n" + "\n".join(f"- `{d}`" for d in self.input_datasets),
            f"\n## Key Features / Outputs",
        ]
        for i, f_ in enumerate(self.key_features, 1):
            lines.append(f"{i}. {f_}")

        lines.append("\n## Phased Build Plan")
        for phase in self.phased_build_plan:
            lines.append(f"\n### {phase.get('phase', '?')}")
            for task in phase.get("tasks", []):
                lines.append(f"- {task}")
            if phase.get("files"):
                lines.append(f"\n  *Output files:* {', '.join(phase['files'])}")

        if self.caveats:
            lines.append("\n## Caveats & Prerequisites")
            for c in self.caveats:
                lines.append(f"- {c}")

        return "\n".join(lines)

    def to_context_md(self) -> str:
        """
        Render spec as a context.md-compatible input for the existing
        pipeline's build phase (run.py pipeline mode).
        """
        features_text = "\n".join(f"- {f}" for f in self.key_features)
        datasets_text = "\n".join(f"- `data/{d}`" for d in self.input_datasets)
        stack_text = "\n".join(f"- {t}" for t in self.tech_stack)
        plan_text = "\n".join(
            f"### {p['phase']}\n" + "\n".join(f"- {t}" for t in p.get("tasks", []))
            for p in self.phased_build_plan
        )

        is_frontend = self.deliverable_type in ("dashboard", "hybrid")

        frontend_contract = ""
        if is_frontend:
            frontend_contract = """
## Frontend Contract (MANDATORY — do not deviate)
The primary deliverable is a **runnable Plotly Dash application**.
These rules are hard requirements, not suggestions:

1. **Entry point**: the app MUST be saved as `pipeline/app.py`.
2. **Launch**: the file must end with `if __name__ == "__main__": app.run(debug=False)`
   so it starts with `python pipeline/app.py`.
3. **User inputs**: the app MUST include at least one interactive input component
   (e.g. `dcc.Dropdown`, `dcc.Input`, `dcc.DatePickerRange`, `dcc.Slider`) that
   controls the analysis or chart output.
4. **Submit / callback**: a button or reactive callback must re-run the underlying
   logic (data load, transform, model/query) and refresh the chart on every change.
   Do NOT pre-compute a static result at startup and display it unchanged.
5. **Charts**: use `plotly.express` or `plotly.graph_objects` rendered inside
   `dcc.Graph` components — NOT static PNG/HTML exports.
6. **No static HTML export**: do not call `fig.write_html()` as the primary output.
   The Dash app IS the deliverable. A CSV export in `outputs/` is fine as a side effect.
7. **Self-contained**: all data loading happens inside the app or its callbacks.
   The app must be runnable with `python pipeline/app.py` from the project root
   with no additional setup beyond `pip install -r requirements.txt`.
8. **Layout**: use `dash_bootstrap_components` or plain Dash HTML components.
   Include a title, an input panel, and a results/chart panel.

Fallback: if Dash app generation fails, still produce a CSV in outputs/ and a
markdown summary — but the Dash app is the primary success criterion.
"""

        return f"""# Build Context — {self.title}

## Business Overview
Auto-generated build context from Strategy Council deliverable recommendation.

## Deliverable Type
{self.deliverable_type}

## Problem Statement
Build {self.title} for {self.target_user}.
{self.rationale}

## Success Criteria
{chr(10).join(f'- Criterion {i+1}: {f}' for i, f in enumerate(self.key_features))}

## Tech Stack
Preferred stack: {', '.join(self.tech_stack)}.
{frontend_contract}
## Build Plan
{plan_text}

## Constraints
- Use only the datasets listed below. Do not invent additional data sources.
- Complexity: {self.estimated_complexity}
- All code outputs go to `pipeline/`, all data outputs go to `outputs/`.

## Caveats
{chr(10).join(f'- {c}' for c in self.caveats) or '- None specified.'}
"""


# ---------------------------------------------------------------------------
# Recommender agent
# ---------------------------------------------------------------------------

_RECOMMENDER_SYSTEM = """\
You are a senior solutions architect. You have read a comprehensive data science
strategy document and must recommend the most appropriate end product (deliverable)
to implement — choosing from:

  - dashboard   : Plotly Dash interactive app (best for exploratory/recurring use)
  - pipeline    : Standalone Python ETL/ML pipeline script (best for batch processing)
  - report      : Automated markdown/HTML report generator (best for periodic summaries)
  - api         : FastAPI REST endpoint (best for real-time prediction serving)
  - notebook    : Jupyter notebook (best for one-off analysis or stakeholder presentations)
  - hybrid      : Combination of two of the above (e.g. pipeline + dashboard)

You must respond with a single JSON object matching EXACTLY this schema:
{{
  "deliverable_type": "dashboard|pipeline|report|api|notebook|hybrid",
  "title": "short descriptive name",
  "rationale": "2-3 sentence justification",
  "target_user": "who uses it and how often",
  "tech_stack": ["list", "of", "libraries"],
  "input_datasets": ["exact filenames from data/ dir that exist"],
  "key_features": ["concrete feature 1", "concrete feature 2", ...],
  "phased_build_plan": [
    {{
      "phase": "Phase 1: Data Preparation",
      "tasks": ["task 1", "task 2"],
      "files": ["outputs/file.py"]
    }}
  ],
  "estimated_complexity": "low|medium|high",
  "caveats": ["caveat 1", "caveat 2"]
}}

Rules:
- input_datasets must only contain files that actually exist in the project data/ directory.
- tech_stack must be Python-based and compatible with the project's existing stack
  (LangGraph, Pandas, Plotly, Dash, FastAPI are all available).
- key_features must be specific and implementable — not vague goals.
- phased_build_plan must have 3-5 phases with concrete tasks.
- Return ONLY valid JSON. No markdown fences, no extra text.

Available files in data/:
{data_files}

Strategy document (read carefully before recommending):
{strategy}
"""


def recommend_deliverable(
    strategy_doc: str,
    data_dir: Path,
    model: str = "gpt-4o",
) -> DeliverableSpec:
    """
    Read the strategy doc and recommend the best end product to build.

    Parameters
    ----------
    strategy_doc : str
        Full text of the strategy markdown document.
    data_dir : Path
        Path to the data/ directory (to list available files).
    model : str
        LLM model to use.

    Returns
    -------
    DeliverableSpec
    """
    # List available data files
    data_files = sorted(
        str(p.relative_to(data_dir.parent))
        for p in data_dir.rglob("*")
        if p.is_file() and not p.name.startswith(".")
    )
    data_files_text = "\n".join(f"  - {f}" for f in data_files)

    logger.info("[DeliverableRecommender] Analysing strategy doc (%d chars)…", len(strategy_doc))

    llm = ChatOpenAI(model=model, temperature=0.2)
    system = _RECOMMENDER_SYSTEM.format(
        data_files=data_files_text,
        strategy=strategy_doc[:12000],  # truncate for context window safety
    )

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content="Produce the deliverable recommendation JSON now."),
    ])

    try:
        raw = json.loads(response.content)
    except (json.JSONDecodeError, TypeError):
        logger.warning("[DeliverableRecommender] JSON parse failed — using fallback spec.")
        raw = _fallback_spec(strategy_doc)

    return DeliverableSpec(
        deliverable_type=raw.get("deliverable_type", "pipeline"),
        title=raw.get("title", "Data Science Pipeline"),
        rationale=raw.get("rationale", ""),
        target_user=raw.get("target_user", "Data team"),
        tech_stack=raw.get("tech_stack", ["Python", "Pandas", "Plotly"]),
        input_datasets=raw.get("input_datasets", []),
        key_features=raw.get("key_features", []),
        phased_build_plan=raw.get("phased_build_plan", []),
        estimated_complexity=raw.get("estimated_complexity", "medium"),
        caveats=raw.get("caveats", []),
    )


def _fallback_spec(strategy_doc: str) -> dict:
    """Minimal fallback if LLM response can't be parsed."""
    return {
        "deliverable_type": "pipeline",
        "title": "Data Science Pipeline",
        "rationale": "Fallback recommendation — manual review of strategy doc recommended.",
        "target_user": "Data team",
        "tech_stack": ["Python", "Pandas", "Plotly", "Dash"],
        "input_datasets": ["copa.csv", "pnl2425_volume_extracts_matched.csv"],
        "key_features": ["Data ingestion and cleaning", "Analysis", "Output generation"],
        "phased_build_plan": [
            {"phase": "Phase 1: Data Prep", "tasks": ["Load and clean data"], "files": []},
            {"phase": "Phase 2: Analysis", "tasks": ["Run analysis"], "files": []},
            {"phase": "Phase 3: Output", "tasks": ["Generate outputs"], "files": []},
        ],
        "estimated_complexity": "medium",
        "caveats": ["Review strategy doc manually before building."],
    }
