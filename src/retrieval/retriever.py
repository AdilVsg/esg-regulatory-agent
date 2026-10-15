"""
Retrieval interface over the Chroma collection built by
chunk_and_embed.py. Used by the agent's retrieve node and by the
evaluation scripts, so the Chroma connection logic lives in one place.
"""
import os

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Read from the environment so the Dockerfile / .env setting is actually
# honoured; the default keeps local runs from the repo root working as before.
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/processed/chroma")
COLLECTION_NAME = "esg_corpus"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        # Telemetry disabled explicitly: Chroma's analytics client crashes on
        # every event with the installed posthog version, polluting the logs.
        _client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)
    return _collection


def retrieve(query, doc_type=None, k=5):
    """
    Retrieve the k most relevant chunks for a query.

    doc_type: None (search everything), "regulatory", or "corporate".
    Returns a list of dicts: {text, source, page, doc_type, distance}.
    """
    collection = _get_collection()
    where = {"doc_type": doc_type} if doc_type else None

    results = collection.query(query_texts=[query], n_results=k, where=where)

    chunks = []
    for text, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        chunks.append({
            "text": text,
            "source": meta["source"],
            "page": meta["page"],
            "doc_type": meta.get("doc_type"),
            "distance": distance,
        })
    return chunks