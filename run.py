#!/usr/bin/env python3
"""Agentic Data Science Pipeline — Main Entry Point.

Orchestrates a multi-agent collaboration pipeline that reads business context
and data source documentation, then designs, builds, tests, and ships a
data science solution.

Usage::

    python run.py \\
        --context inputs/sample/context.md \\
        --data-sources inputs/sample/data_sources.md \\
        --quality balanced \\
        --samples-dir inputs/sample/

Run ``python run.py --help`` for full options.
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

# Ensure the project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from agents.persona_factory import PersonaFactory
from agents.swarm import Phase, SwarmState, build_graph
from mvp_bundle import generate_projection_bundle

console = Console()

# ======================================================================
# Configuration
# ======================================================================


def load_config(config_path: Path, quality_override: str | None = None) -> dict[str, Any]:
    """Load config.yaml and apply the selected quality preset.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file.
    quality_override : str | None
        If given, overrides the ``active_preset`` key in the file.

    Returns
    -------
    dict[str, Any]
        Merged configuration dictionary with the active preset's settings
        promoted to top-level ``model_config`` and ``workflow`` keys.
    """
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    preset_name = quality_override or config.get("active_preset", "balanced")
    presets = config.get("quality_presets", {})

    if preset_name not in presets:
        console.print(f"[red]Unknown quality preset '{preset_name}'. Using 'balanced'.[/]")
        preset_name = "balanced"

    preset = presets[preset_name]
    config["active_preset"] = preset_name
    config["model_config"] = preset["model_config"]
    config["workflow"] = preset["workflow"]

    return config


def read_file(path: str | Path) -> str:
    """Read a text file and return its contents."""
    p = Path(path)
    if not p.exists():
        console.print(f"[red]File not found: {p}[/]")
        sys.exit(1)
    return p.read_text(encoding="utf-8")


# ======================================================================
# Display helpers
# ======================================================================


def display_banner(config: dict[str, Any]) -> None:
    """Show a startup banner with configuration summary."""
    preset = config["active_preset"]
    workflow = config["workflow"]

    table = Table(title="Pipeline Configuration", border_style="blue")
    table.add_column("Setting", style="bold")
    table.add_column("Value")
    table.add_row("Quality Preset", preset)
    table.add_row("Max Review Loops", str(workflow["max_review_loops"]))
    table.add_row("Require Unanimous", str(workflow["require_unanimous"]))
    table.add_row("Reflection Enabled", str(workflow["reflection_enabled"]))
    table.add_row("Best Practice Checks", workflow["best_practice_check_frequency"])

    console.print()
    console.print(
        Panel(
            "[bold blue]Agentic Data Science Pipeline[/]\n"
            "Multi-agent collaboration for data science solutions",
            border_style="blue",
        )
    )
    console.print(table)
    console.print()


def display_personas(personas: list[Any]) -> None:
    """Show the selected agent roster."""
    table = Table(title="Agent Roster", border_style="green")
    table.add_column("Agent", style="bold")
    table.add_column("Type")
    table.add_column("Model")
    table.add_column("Domain Focus")

    for p in personas:
        table.add_row(
            p.name,
            p.config.persona_type,
            p.config.model,
            p.config.domain_focus or "—",
        )

    console.print(table)
    console.print()


# ======================================================================
# Git helpers
# ======================================================================


def git_init_and_commit(message: str) -> None:
    """Initialise a git repo (if needed) and commit all files.

    Parameters
    ----------
    message : str
        The commit message.
    """
    try:
        if not (_PROJECT_ROOT / ".git").exists():
            subprocess.run(["git", "init"], cwd=_PROJECT_ROOT, check=True, capture_output=True)
            logger.info("Initialised git repository.")

        subprocess.run(["git", "add", "-A"], cwd=_PROJECT_ROOT, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=_PROJECT_ROOT,
            check=True,
            capture_output=True,
        )
        logger.info("Committed: %s", message)
        console.print(f"[green]Git commit:[/] {message}")
    except FileNotFoundError:
        logger.warning("git not found — skipping commit.")
    except subprocess.CalledProcessError as exc:
        logger.warning("git command failed: %s", exc)


def _find_available_port(preferred_port: int, max_tries: int = 50) -> int:
    """Return the first available localhost port from preferred_port upward."""
    for port in range(preferred_port, preferred_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(
        f"No available port found from {preferred_port} to {preferred_port + max_tries - 1}"
    )


# ======================================================================
# Main pipeline
# ======================================================================


def run_pipeline(
    context_path: str,
    data_sources_path: str,
    quality: str | None,
    samples_dir: str | None,
    config_path: str = "config.yaml",
    launch_frontend: bool = False,
) -> None:
    """Execute the full agentic pipeline.

    Parameters
    ----------
    context_path : str
        Path to the business context markdown file.
    data_sources_path : str
        Path to the data sources documentation file.
    quality : str | None
        Quality preset override (``fast``, ``balanced``, ``maximum``).
    samples_dir : str | None
        Optional directory containing sample data files for profiling.
    config_path : str
        Path to the YAML configuration file.
    launch_frontend : bool
        If True, start the Dash frontend after a successful pipeline run.
    """
    # Load environment and config
    load_dotenv()
    config = load_config(Path(config_path), quality_override=quality)
    display_banner(config)

    # Read input files
    context_text = read_file(context_path)
    data_sources_text = read_file(data_sources_path)

    console.print(f"[dim]Context loaded: {len(context_text):,} chars[/]")
    console.print(f"[dim]Data sources loaded: {len(data_sources_text):,} chars[/]")
    console.print()

    # ------------------------------------------------------------------
    # Persona Factory — decide which agents to spin up
    # ------------------------------------------------------------------
    console.rule("[bold]Phase 0 — Agent Selection")
    factory = PersonaFactory(
        factory_config=config.get("persona_factory", {}),
        model_config=config["model_config"],
    )
    personas = factory.decide_personas(context_text, data_sources_text)
    display_personas(personas)

    # ------------------------------------------------------------------
    # Build the LangGraph state machine
    # ------------------------------------------------------------------
    workflow_cfg = config["workflow"]
    graph = build_graph(personas, workflow_cfg)

    # Prepare initial state
    initial_state = SwarmState(
        context_text=context_text,
        data_sources_text=data_sources_text,
        samples_dir=samples_dir or "",
        max_review_loops=workflow_cfg["max_review_loops"],
        require_unanimous=workflow_cfg["require_unanimous"],
        reflection_enabled=workflow_cfg["reflection_enabled"],
        best_practice_check_frequency=workflow_cfg["best_practice_check_frequency"],
    )

    # ------------------------------------------------------------------
    # Execute the graph
    # ------------------------------------------------------------------
    console.rule("[bold]Executing Pipeline")
    console.print()

    try:
        final_state = graph.invoke(initial_state.model_dump())
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user.[/]")
        return

    # ------------------------------------------------------------------
    # Phase D — Ship
    # ------------------------------------------------------------------
    final_phase = final_state.get("current_phase", Phase.DONE)
    if final_phase == Phase.ABORTED or final_phase == Phase.ABORTED.value:
        console.print(Panel("[bold red]Pipeline aborted by reviewer.[/]", border_style="red"))
        return

    console.rule("[bold green]Phase D — Ship")

    # Deterministic MVP artefacts (chart + csv + report) so each run
    # produces tangible deliverables even if agent tool-calling is limited.
    try:
        artefact_paths = generate_projection_bundle(
            data_dir=samples_dir or (_PROJECT_ROOT / "data"),
            output_dir=_PROJECT_ROOT / "outputs",
        )
        console.print("[green]Generated MVP artefacts:[/]")
        for name, path in artefact_paths.items():
            console.print(f"  - {name}: {path.relative_to(_PROJECT_ROOT)}")
    except Exception as exc:
        logger.warning("MVP artefact generation skipped: %s", exc)
        console.print(f"[yellow]MVP artefact generation skipped:[/] {exc}")

    # Save run snapshot to ChromaDB agent memory
    try:
        from store.memory import save_run_snapshot
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_run_snapshot(run_id=run_timestamp, final_state=final_state, config=config)
    except Exception as exc:
        logger.warning("Memory snapshot skipped: %s", exc)

    # Commit everything to git
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    git_init_and_commit(f"Pipeline run completed at {timestamp}")

    # Summary
    artifacts = final_state.get("artifacts", [])
    console.print(
        Panel(
            f"[bold green]Pipeline completed successfully![/]\n\n"
            f"Artefacts produced: {len(artifacts)}\n"
            f"Review loops: {final_state.get('review_count', 0)}\n"
            f"Quality preset: {config['active_preset']}\n\n"
            "Check the [bold]outputs/[/] and [bold]pipeline/[/] directories.",
            title="Summary",
            border_style="green",
        )
    )

    if launch_frontend:
        app_path = _PROJECT_ROOT / "pipeline" / "launch_projection_dash_app.py"
        if app_path.exists():
            preferred_port = 8050
            try:
                port = _find_available_port(preferred_port)
                env = os.environ.copy()
                env["DASH_PORT"] = str(port)
                env["DASH_DEBUG"] = "0"
                console.print(f"[cyan]Launching Dash frontend on localhost:{port}...[/]")
                subprocess.Popen([sys.executable, str(app_path)], cwd=_PROJECT_ROOT, env=env)
                console.print(f"[green]Frontend started:[/] http://127.0.0.1:{port}")
            except Exception as exc:
                logger.warning("Failed to launch frontend: %s", exc)
                console.print(f"[yellow]Failed to launch frontend:[/] {exc}")
        else:
            console.print("[yellow]Frontend script not found; skipping launch.[/]")


# ======================================================================
# CLI
# ======================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with subcommands."""
    parser = argparse.ArgumentParser(
        description="Agentic Data Science Pipeline — Multi-agent collaboration system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modes:\n"
            "  pipeline   Run the full build pipeline (understand → build → test → ship)\n"
            "  strategy   Run the Strategy Council (enrich → debate → critique → synthesise)\n"
            "\nExamples:\n"
            "  python run.py pipeline --context inputs/sample/context.md "
            "--data-sources inputs/sample/data_sources.md\n"
            "  python run.py pipeline --context inputs/sample/context.md "
            "--data-sources inputs/sample/data_sources.md --quality maximum\n"
            "  python run.py strategy --problem inputs/sample/context.md "
            "--datasets copa pnl launch_tracker\n"
            "  python run.py strategy --problem inputs/sample/context.md "
            "--datasets copa pnl --quality balanced --verbose\n"
        ),
    )

    # ── Shared flags (available to all subcommands) ──────────────────────
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--quality",
        choices=["fast", "balanced", "maximum"],
        default=None,
        help="Quality preset (overrides config.yaml). Default: balanced.",
    )
    shared.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the configuration YAML file. Default: config.yaml.",
    )
    shared.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )

    subparsers = parser.add_subparsers(dest="mode", metavar="MODE")
    subparsers.required = True

    # ── pipeline subcommand (existing behaviour, unchanged) ──────────────
    sp_pipeline = subparsers.add_parser(
        "pipeline",
        parents=[shared],
        help="Run the full multi-agent build pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sp_pipeline.add_argument(
        "--context",
        required=True,
        help="Path to the business context markdown file.",
    )
    sp_pipeline.add_argument(
        "--data-sources",
        required=True,
        help="Path to the data sources documentation markdown file.",
    )
    sp_pipeline.add_argument(
        "--samples-dir",
        default=None,
        help="Optional directory containing sample data files for profiling.",
    )
    sp_pipeline.add_argument(
        "--launch-frontend",
        action="store_true",
        help="Launch Dash frontend on localhost after successful completion.",
    )

    # ── strategy subcommand (new: Strategy Council) ───────────────────────
    sp_strategy = subparsers.add_parser(
        "strategy",
        parents=[shared],
        help="Run the Strategy Council: 3 agents + 2 critic rounds + web search.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Reference specific datasets by name (matches filenames in data/)\n"
            "  python run.py strategy --problem inputs/sample/context.md "
            "--datasets copa pnl launch_tracker\n"
            "\n"
            "  # Pass a problem statement inline instead of a file\n"
            "  python run.py strategy "
            "--problem-text \"Forecast 12-month SKU sales for SEA markets\" "
            "--datasets copa tm1 pnl\n"
        ),
    )
    sp_strategy.add_argument(
        "--problem",
        default=None,
        metavar="FILE",
        help="Path to a markdown file describing the business problem "
             "(e.g. inputs/sample/context.md). Mutually exclusive with --problem-text.",
    )
    sp_strategy.add_argument(
        "--problem-text",
        default=None,
        metavar="TEXT",
        help="Inline business problem statement string. "
             "Mutually exclusive with --problem.",
    )
    sp_strategy.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        metavar="DATASET",
        help=(
            "One or more dataset names to reference. Use short names that "
            "match filenames in data/ (e.g. copa pnl launch_tracker tm1 iqvia). "
            "The council will look up their full descriptions from ChromaDB "
            "and data_sources.md automatically."
        ),
    )
    sp_strategy.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Optional path to write the final strategy markdown. "
             "Default: outputs/strategy/<run_id>_strategy.md",
    )
    sp_strategy.add_argument(
        "--no-web-search",
        action="store_true",
        help="Disable web search (useful if TAVILY_API_KEY is not set).",
    )

    return parser.parse_args()


# ======================================================================
# Entry point
# ======================================================================

# Module-level logger — configured in __main__ block
logger = logging.getLogger("agentic_ds")

if __name__ == "__main__":
    args = parse_args()

    # Configure logging with Rich
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    logger = logging.getLogger("agentic_ds")

    # Change to project root so relative paths work
    os.chdir(_PROJECT_ROOT)

    # ── Route to the correct mode ────────────────────────────────────────

    if args.mode == "pipeline":
        # Existing pipeline behaviour — unchanged
        run_pipeline(
            context_path=args.context,
            data_sources_path=args.data_sources,
            quality=args.quality,
            samples_dir=args.samples_dir,
            config_path=args.config,
            launch_frontend=args.launch_frontend,
        )

    elif args.mode == "strategy":
        from orchestration.strategy_council import run_strategy_council
        from pathlib import Path as _Path

        # Resolve problem statement
        if args.problem and args.problem_text:
            console.print("[red]Use either --problem or --problem-text, not both.[/]")
            sys.exit(1)
        if args.problem:
            problem_statement = _Path(args.problem).read_text(encoding="utf-8")
        elif args.problem_text:
            problem_statement = args.problem_text
        else:
            console.print("[red]Provide --problem <file> or --problem-text <text>.[/]")
            sys.exit(1)

        # Resolve dataset descriptions from short names
        dataset_map = {
            "copa":           "copa.csv — historical P&L actuals (PRODUCT_ID, PERIOD, COUNTRY_CODE, SALES_QUANTITY, REVENUE_GOODS)",
            "pnl":            "pnl2425_volume_extracts_matched.csv — FY24-25 P&L and volume forecasts (matched_SKU_ID, Market, forecast_volume_y1, forecast_net_sales_y1)",
            "launch_tracker": "launch_tracker25_matched.csv — SKU launch history (SKU Code, SKU Launch Month, Category, Market Specific, Brand)",
            "tm1":            "tm1_qty_sales_pivot.csv — monthly actual quantity and net sales (sku_id, period, quantity, net_sales)",
            "iqvia":          "IQVIA_Asia_data1.csv — third-party Asia market size data",
            "euromonitor":    "euro_mon_hier1_RSP_USD_histconst2024_histfixedER20242.csv — Euromonitor retail selling price data",
            "nicholas_hall":  "Nicholas_Hall.csv — OTC market intelligence data",
            "who_flu":        "WHO_FLU.csv — WHO flu surveillance data",
            "forecast":       "data/raw/forecast.csv — raw forecast extracts",
        }
        dataset_descriptions = [
            dataset_map.get(d.lower().replace("-", "_"), d)
            for d in args.datasets
        ]

        # Load config for preset
        config = load_config(_Path(args.config), quality_override=args.quality)
        preset = config.get("active_preset", "balanced")

        # Output path
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = _Path(args.output) if args.output else \
            _Path(f"outputs/strategy/{run_ts}_strategy.md")

        console.print()
        console.print(
            Panel(
                f"[bold blue]Strategy Council[/]\n"
                f"Datasets: {', '.join(args.datasets)}\n"
                f"Quality preset: {preset}\n"
                f"Web search: {'disabled' if args.no_web_search else 'enabled'}\n"
                f"Output: {output_path}",
                border_style="blue",
            )
        )
        console.print()

        # Disable web search if requested
        if args.no_web_search:
            import os as _os
            _os.environ["TAVILY_API_KEY"] = ""  # tools will skip gracefully

        result = run_strategy_council(
            problem_statement=problem_statement,
            dataset_descriptions=dataset_descriptions,
            quality_preset=preset,
            run_id=run_ts,
            output_path=output_path,
        )

        console.print(
            Panel(
                f"[bold green]Strategy Council complete![/]\n\n"
                f"Output written to: [bold]{output_path}[/]\n"
                f"Critique rounds: {len(result['critiques'])}\n"
                f"Agent proposals: {len(result['proposals'])}\n"
                f"Context chunks used: {len(result['retrieved_chunks'])}",
                title="Summary",
                border_style="green",
            )
        )
        console.print(f"\n[dim]Preview (first 600 chars):[/]")
        console.print(result["final_strategy"][:600])
