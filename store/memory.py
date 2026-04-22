"""
store/memory.py — Write pipeline run outputs into agent_memory collection.

Called automatically at the end of each pipeline run (Phase D — Ship) to
snapshot agent decisions, outputs, and ADRs for future retrieval.

Usage in run.py (after final_state is produced):
    from store.memory import save_run_snapshot
    save_run_snapshot(run_id=timestamp, final_state=final_state, config=config)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from store.client import get_collection
from store.config import COLLECTION_AGENT_MEMORY
from store.ingest import _doc_id

logger = logging.getLogger(__name__)


def save_run_snapshot(
    run_id: str,
    final_state: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> None:
    """
    Persist a pipeline run snapshot to the agent_memory collection.

    Parameters
    ----------
    run_id : str
        Unique run identifier (e.g. ISO timestamp from run.py).
    final_state : dict
        The final SwarmState dict after graph.invoke().
    config : dict | None
        The loaded config.yaml dict (for preset / model labelling).
    """
    collection = get_collection(COLLECTION_AGENT_MEMORY)

    agent_outputs = final_state.get("agent_outputs", {})
    artifacts = final_state.get("artifacts", [])
    feedback_log = final_state.get("feedback_log", [])
    review_count = final_state.get("review_count", 0)
    phase = final_state.get("current_phase", "unknown")
    preset = config.get("active_preset", "unknown") if config else "unknown"

    # Build a human-readable snapshot text
    outputs_text = "\n\n".join(
        f"### {name}\n{output[:1000]}"
        for name, output in agent_outputs.items()
        if not name.endswith("_review")
    )

    snapshot_text = f"""# Pipeline Run Snapshot
Run ID: {run_id}
Preset: {preset}
Final Phase: {phase}
Review Loops: {review_count}
Artifacts: {', '.join(artifacts) or 'none'}

## Agent Outputs (truncated)
{outputs_text or '(none)'}

## Feedback Log
{chr(10).join(f'- {fb}' for fb in feedback_log) or '(none)'}
"""

    doc_id = _doc_id("agent_memory", run_id)
    collection.upsert(
        ids=[doc_id],
        documents=[snapshot_text],
        metadatas=[{
            "run_id": run_id,
            "preset": preset,
            "phase": str(phase),
            "review_count": review_count,
            "timestamp": datetime.utcnow().isoformat(),
        }],
    )

    logger.info("[Memory] Saved run snapshot: %s", run_id)
