"""
store/client.py — Singleton ChromaDB client.

Import `get_client()` from anywhere in the project to get the shared,
persistent ChromaDB instance. Collections are created on first access.
"""

from __future__ import annotations

import chromadb
from chromadb.config import Settings

from store.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL, USE_OPENAI_EMBEDDINGS

_client: chromadb.PersistentClient | None = None


def get_client() -> chromadb.PersistentClient:
    """Return the singleton ChromaDB persistent client."""
    global _client
    if _client is None:
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(CHROMA_PERSIST_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def get_embedding_function():
    """Return the configured embedding function."""
    if USE_OPENAI_EMBEDDINGS:
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
        import os
        return OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name="text-embedding-3-small",
        )
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def get_collection(name: str):
    """Get or create a named collection with the configured embedding function."""
    client = get_client()
    ef = get_embedding_function()
    return client.get_or_create_collection(name=name, embedding_function=ef)
