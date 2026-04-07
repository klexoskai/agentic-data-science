"""Human review gate — pauses the pipeline for stakeholder approval.

Displays generated artefacts using Rich for attractive console output, then
waits for the user to approve, reject, or provide feedback.  Each significant
decision is logged as an Architecture Decision Record (ADR).
"""

from __future__ import annotations

import logging
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DECISIONS_DIR = _PROJECT_ROOT / "decisions"
_OUTPUTS_DIR = _PROJECT_ROOT / "outputs"

console = Console()


class ReviewVerdict(str, Enum):
    """Possible outcomes of a human review."""

    APPROVE = "approve"
    REJECT = "reject"
    FEEDBACK = "feedback"


class HumanReviewGate:
    """Interactive gate that presents artefacts and collects human feedback.

    Parameters
    ----------
    gate_name : str
        A short label for this gate (e.g. ``"Gate 1 — Architecture Review"``).
    """

    _adr_counter: int = 0  # class-level counter for ADR numbering

    def __init__(self, gate_name: str) -> None:
        self.gate_name = gate_name

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------
    def _display_artefact(self, title: str, content: str, content_type: str = "markdown") -> None:
        """Render a single artefact in the console."""
        if content_type == "python":
            renderable = Syntax(content, "python", theme="monokai", line_numbers=True)
        elif content_type == "mermaid":
            renderable = Syntax(content, "text", theme="monokai")
        else:
            renderable = Markdown(content)

        console.print(Panel(renderable, title=f"[bold cyan]{title}[/]", border_style="cyan"))

    def _display_summary_table(self, agent_outputs: dict[str, str]) -> None:
        """Show a summary table of what each agent decided."""
        table = Table(title="Agent Decisions Summary", border_style="blue")
        table.add_column("Agent", style="bold")
        table.add_column("Output Preview", max_width=80)

        for agent_name, output in agent_outputs.items():
            preview = output[:200].replace("\n", " ") + ("…" if len(output) > 200 else "")
            table.add_row(agent_name, preview)

        console.print(table)

    # ------------------------------------------------------------------
    # Core gate method
    # ------------------------------------------------------------------
    def run(
        self,
        artefacts: dict[str, str],
        agent_outputs: dict[str, str],
        artefact_types: dict[str, str] | None = None,
    ) -> tuple[ReviewVerdict, str]:
        """Execute the gate: display artefacts, wait for human input.

        Parameters
        ----------
        artefacts : dict[str, str]
            Mapping of artefact title → content to display.
        agent_outputs : dict[str, str]
            Mapping of agent name → their summary output for the table.
        artefact_types : dict[str, str] | None
            Optional mapping of artefact title → content type
            (``"markdown"``, ``"python"``, ``"mermaid"``).

        Returns
        -------
        tuple[ReviewVerdict, str]
            The verdict and any feedback text (empty string if approved).
        """
        artefact_types = artefact_types or {}

        console.print()
        console.rule(f"[bold yellow] {self.gate_name} [/]")
        console.print()

        # Display each artefact
        for title, content in artefacts.items():
            ctype = artefact_types.get(title, "markdown")
            self._display_artefact(title, content, content_type=ctype)
            console.print()

        # Display summary table
        self._display_summary_table(agent_outputs)
        console.print()

        # Save artefacts to outputs/ for reference
        for title, content in artefacts.items():
            safe_name = title.lower().replace(" ", "_").replace("/", "_")
            ext = ".py" if artefact_types.get(title) == "python" else ".md"
            if artefact_types.get(title) == "mermaid":
                ext = ".mmd"
            out_path = _OUTPUTS_DIR / f"{safe_name}{ext}"
            _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content, encoding="utf-8")
            logger.info("Saved artefact to %s", out_path)

        # Prompt for decision
        console.print(
            Panel(
                "[bold]Options:[/]\n"
                "  [green]y[/] — Approve and continue\n"
                "  [red]n[/] — Reject and stop\n"
                "  [yellow]f[/] — Provide feedback (agents will revise)",
                title="Your Decision",
                border_style="yellow",
            )
        )

        while True:
            choice = console.input("[bold]Your choice (y/n/f): [/]").strip().lower()
            if choice in ("y", "yes"):
                verdict = ReviewVerdict.APPROVE
                feedback = ""
                break
            elif choice in ("n", "no"):
                verdict = ReviewVerdict.REJECT
                feedback = console.input("[bold red]Reason for rejection: [/]").strip()
                break
            elif choice in ("f", "feedback"):
                feedback = console.input("[bold yellow]Your feedback: [/]").strip()
                verdict = ReviewVerdict.FEEDBACK
                break
            else:
                console.print("[red]Invalid choice. Enter y, n, or f.[/]")

        # Log the decision as an ADR
        self._write_adr(
            title=self.gate_name,
            context=f"Pipeline paused at {self.gate_name} for human review.",
            decision=f"Reviewer verdict: {verdict.value}",
            consequences="Pipeline continues." if verdict == ReviewVerdict.APPROVE else "Agents will revise based on feedback.",
            feedback=feedback,
        )

        console.print()
        if verdict == ReviewVerdict.APPROVE:
            console.print("[bold green]✓ Approved — continuing pipeline.[/]")
        elif verdict == ReviewVerdict.REJECT:
            console.print("[bold red]✗ Rejected — pipeline stopped.[/]")
        else:
            console.print("[bold yellow]↻ Feedback received — agents will revise.[/]")
        console.print()

        return verdict, feedback

    # ------------------------------------------------------------------
    # ADR logging
    # ------------------------------------------------------------------
    def _write_adr(
        self,
        title: str,
        context: str,
        decision: str,
        consequences: str,
        feedback: str,
    ) -> Path:
        """Write an Architecture Decision Record to the decisions/ directory.

        Returns
        -------
        Path
            Path to the written ADR file.
        """
        HumanReviewGate._adr_counter += 1
        num = HumanReviewGate._adr_counter

        _DECISIONS_DIR.mkdir(parents=True, exist_ok=True)

        adr_content = (
            f"# ADR-{num:04d}: {title}\n\n"
            f"## Status\nAccepted\n\n"
            f"## Date\n{date.today().isoformat()}\n\n"
            f"## Context\n{context}\n\n"
            f"## Decision\n{decision}\n\n"
            f"## Consequences\n{consequences}\n\n"
            f"## Feedback\n{feedback if feedback else 'None'}\n"
        )

        adr_path = _DECISIONS_DIR / f"adr-{num:04d}.md"
        adr_path.write_text(adr_content, encoding="utf-8")
        logger.info("Written ADR: %s", adr_path)
        return adr_path
