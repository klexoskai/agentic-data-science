"""
gates/strategy_gate.py — Strategy Gate (Gate 0).

Sits between:
    Strategy Council (produces strategy .md)
        ↓
    [Gate 0 — Strategy Approval]  ← YOU ARE HERE
        ↓
    Pipeline build phase

Displays to the user:
  1. The strategy document summary (first N lines)
  2. The Deliverable Recommender's spec (type, features, build plan)

User choices:
  a  Approve     → proceed to build with the recommended spec
  r  Revise      → re-run Strategy Council with additional feedback
  s  Skip build  → save the strategy doc only, do not build
  q  Quit        → abort everything
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


class StrategyVerdict(str, Enum):
    APPROVE = "approve"
    REVISE = "revise"
    SKIP_BUILD = "skip_build"
    QUIT = "quit"


def run_strategy_gate(
    strategy_doc: str,
    strategy_path: Path,
    deliverable_spec_md: str,
    deliverable_spec,           # DeliverableSpec instance
) -> tuple[StrategyVerdict, str]:
    """
    Display the strategy doc + deliverable recommendation and prompt for approval.

    Parameters
    ----------
    strategy_doc : str
        Full text of the strategy markdown.
    strategy_path : Path
        Path where the strategy .md was saved (shown to user).
    deliverable_spec_md : str
        Rendered markdown of the DeliverableSpec.
    deliverable_spec : DeliverableSpec
        The structured spec (used for summary table).

    Returns
    -------
    (StrategyVerdict, feedback_text)
        feedback_text is populated when verdict == REVISE.
    """
    console.print()
    console.rule("[bold blue]Gate 0 — Strategy Approval[/]", style="blue")
    console.print()

    # ── 1. Strategy doc summary ─────────────────────────────────────────────
    console.print(Panel(
        f"[bold]Strategy document saved:[/] {strategy_path}\n\n"
        + _truncate_preview(strategy_doc, lines=30),
        title="[bold blue]Strategy Document (preview)",
        border_style="blue",
        expand=False,
    ))
    console.print()

    # ── 2. Deliverable recommendation ───────────────────────────────────────
    _print_deliverable_summary(deliverable_spec)
    console.print()

    console.print(Panel(
        deliverable_spec_md,
        title="[bold cyan]Deliverable Recommendation (full spec)",
        border_style="cyan",
        expand=False,
    ))
    console.print()

    # ── 3. Prompt ────────────────────────────────────────────────────────────
    console.print(
        "[bold]What would you like to do?[/]\n"
        "  [green]a[/]  Approve — proceed to build\n"
        "  [yellow]r[/]  Revise  — re-run strategy council with feedback\n"
        "  [dim]s[/]  Skip    — save strategy doc only, no build\n"
        "  [red]q[/]  Quit    — abort\n"
    )

    while True:
        choice = Prompt.ask(
            "[bold]Your choice[/]",
            choices=["a", "r", "s", "q"],
            default="a",
        ).strip().lower()

        if choice == "a":
            console.print("[green]✓ Approved — proceeding to build.[/]")
            return StrategyVerdict.APPROVE, ""

        elif choice == "r":
            feedback = Prompt.ask(
                "[yellow]Enter feedback for the Strategy Council[/] "
                "(what to reconsider or add)"
            ).strip()
            if feedback:
                console.print(f"[yellow]Feedback recorded. Re-running council…[/]")
                return StrategyVerdict.REVISE, feedback
            else:
                console.print("[dim]No feedback entered — please try again.[/]")
                continue

        elif choice == "s":
            console.print(
                f"[dim]Build skipped. Strategy doc at: {strategy_path}[/]"
            )
            return StrategyVerdict.SKIP_BUILD, ""

        elif choice == "q":
            console.print("[red]Aborted.[/]")
            return StrategyVerdict.QUIT, ""


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_deliverable_summary(spec) -> None:
    """Print a Rich table summarising the deliverable spec."""
    table = Table(
        title="Recommended End Product",
        border_style="cyan",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Property", style="bold", width=22)
    table.add_column("Value")

    table.add_row("Type",        f"[bold]{spec.deliverable_type}[/]")
    table.add_row("Title",       spec.title)
    table.add_row("Target user", spec.target_user)
    table.add_row("Complexity",  _complexity_badge(spec.estimated_complexity))
    table.add_row("Tech stack",  ", ".join(spec.tech_stack))
    table.add_row("Datasets",    "\n".join(spec.input_datasets))
    table.add_row(
        "Key features",
        "\n".join(f"{i+1}. {f}" for i, f in enumerate(spec.key_features[:5]))
    )
    table.add_row(
        "Build phases",
        "\n".join(p.get("phase", "") for p in spec.phased_build_plan)
    )

    console.print(table)


def _complexity_badge(complexity: str) -> str:
    colors = {"low": "green", "medium": "yellow", "high": "red"}
    c = colors.get(complexity.lower(), "white")
    return f"[{c}]{complexity}[/{c}]"


def _truncate_preview(text: str, lines: int = 30) -> str:
    all_lines = text.splitlines()
    preview = "\n".join(all_lines[:lines])
    if len(all_lines) > lines:
        preview += f"\n\n[dim]… ({len(all_lines) - lines} more lines — open {{}}) …[/dim]"
    return preview
