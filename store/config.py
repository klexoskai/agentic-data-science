"""
store/config.py — Central config for the ChromaDB context store.

All paths and collection names are defined here so every module
references a single source of truth.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where ChromaDB persists its on-disk data (gitignored)
CHROMA_PERSIST_DIR = PROJECT_ROOT / "store" / ".chroma"

# Data sources
DATA_DIR = PROJECT_ROOT / "data"
INPUTS_DIR = PROJECT_ROOT / "inputs"

# ── Collections ────────────────────────────────────────────────────────────
# One collection per context type — keeps retrieval focused.

COLLECTION_DATA_SOURCES = "data_sources"
# Chunked rows from CSV files (COPA, TM1, P&L, Launch Tracker, IQVIA, etc.)
# Chunk strategy: group by SKU + period window (~50-200 rows per chunk)

COLLECTION_CONTEXT_DOCS = "context_docs"
# Markdown files: context.md, data_sources.md, README, ADRs
# Chunk strategy: split by heading (##) — each section is one chunk

COLLECTION_AGENT_MEMORY = "agent_memory"
# Past agent outputs, pipeline run summaries, Gate decisions
# Chunk strategy: one document per pipeline run

ALL_COLLECTIONS = [
    COLLECTION_DATA_SOURCES,
    COLLECTION_CONTEXT_DOCS,
    COLLECTION_AGENT_MEMORY,
]

# ── Embedding model ─────────────────────────────────────────────────────────
# Uses sentence-transformers locally — no API key needed.
# Swap to "text-embedding-3-small" (OpenAI) by changing EMBEDDING_MODEL
# and setting USE_OPENAI_EMBEDDINGS = True.

EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # fast, 384-dim, good for English text
USE_OPENAI_EMBEDDINGS = False

# ── Retrieval defaults ──────────────────────────────────────────────────────

DEFAULT_N_RESULTS = 5          # chunks returned per query
MAX_CHUNK_CHARS = 2000         # max characters per chunk before splitting
