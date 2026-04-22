# Context Store — ChromaDB

Centralised vector store for all agent context: CSV data, markdown docs, and past run memory.

## Structure

```
store/
  config.py      — paths, collection names, embedding model config
  client.py      — singleton ChromaDB client (PersistentClient)
  ingest.py      — ingestion pipeline (CSVs + markdown docs)
  retriever.py   — query interface used by all agent nodes
  memory.py      — writes pipeline run snapshots for future retrieval
  .chroma/       — on-disk ChromaDB data (gitignored)
  README.md      — this file
```

## Collections

| Collection | Contents | Chunk strategy |
|---|---|---|
| `data_sources` | CSV rows from all data files | Grouped by SKU, or row batches for large files |
| `context_docs` | Markdown files (context.md, data_sources.md, ADRs) | Split by `##` heading |
| `agent_memory` | Past pipeline run snapshots | One document per run |

## Setup

```bash
pip install chromadb sentence-transformers
```

## Ingest everything

```bash
# From project root
python -m store.ingest                  # ingest all sources
python -m store.ingest --source csv     # only CSV data files
python -m store.ingest --source docs    # only markdown docs
python -m store.ingest --reset          # wipe + re-ingest
```

Re-run ingest whenever you update your data files or context docs.
It uses `upsert` so it's safe to run repeatedly — only changed chunks are rewritten.

## Query from Python

```python
from store.retriever import retrieve, retrieve_by_sku, retrieve_for_query

# General semantic search across all collections
chunks = retrieve("SEA revenue trend Q2 2025")

# SKU-specific lookup
chunks = retrieve_by_sku("105905")

# Full pipeline retrieval (used by AIQ research graph)
chunks = retrieve_for_query(
    query="What is the YTD performance of Betadine in Australia?",
    context_text=context_text,
    data_sources_text=data_sources_text,
)

# Each chunk: {"text": str, "source": str, "metadata": dict, "distance": float}
for chunk in chunks:
    print(chunk["source"], "—", chunk["text"][:120])
```

## Switching to OpenAI embeddings

In `store/config.py`:
```python
USE_OPENAI_EMBEDDINGS = True
EMBEDDING_MODEL = "text-embedding-3-small"   # ignored when USE_OPENAI_EMBEDDINGS=True
```
Make sure `OPENAI_API_KEY` is set in your `.env`.

## Integration with AIQ research graph

The `orchestration/nodes.py` and `integration/pipeline_bridge.py` modules call
`retrieve_for_query()` automatically. No changes needed — just run ingest first.

## Integration with existing run.py (agent memory)

Add to the end of `run_pipeline()` in `run.py`, after `final_state` is produced:

```python
from store.memory import save_run_snapshot
save_run_snapshot(run_id=timestamp, final_state=final_state, config=config)
```
