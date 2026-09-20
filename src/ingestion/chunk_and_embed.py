"""
Chunk the extracted per-page text and embed it into a persistent Chroma
collection.

Run after extract_corpus.py. Safe to re-run: the collection is deleted
and rebuilt from scratch each time so ingestion stays reproducible.
"""
import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter

REGULATORY_SOURCES = {
    "01_csrd_directive_2022_2464.pdf",
    "02_esrs_set1_2023_2772.pdf",
    "03_taxonomy_regulation_2020_852.pdf",
}

TEXT_DIR = Path("data/processed/text")
CHROMA_DIR = Path("data/processed/chroma")
COLLECTION_NAME = "esg_corpus"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BATCH_SIZE = 500


def load_pages():
    pages = []
    for path in sorted(TEXT_DIR.glob("*.json")):
        if path.name == "_manifest.json":
            continue
        with open(path, encoding="utf-8") as f:
            pages.extend(json.load(f))
    return pages


def main():
    pages = load_pages()
    print(f"Loaded {len(pages)} pages from {TEXT_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents, metadatas, ids = [], [], []
    for page in pages:
        chunks = splitter.split_text(page["text"])
        doc_type = "regulatory" if page["source"] in REGULATORY_SOURCES else "corporate"
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"source": page["source"], "page": page["page"], "doc_type": doc_type})
            ids.append(f"{page['source']}_p{page['page']}_c{i}")

    print(f"Split into {len(documents)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

    for i in range(0, len(documents), BATCH_SIZE):
        collection.add(
            documents=documents[i:i + BATCH_SIZE],
            metadatas=metadatas[i:i + BATCH_SIZE],
            ids=ids[i:i + BATCH_SIZE],
        )
        print(f"  embedded batch {i}-{min(i + BATCH_SIZE, len(documents))}")

    print(f"\nDone. {collection.count()} chunks stored in {CHROMA_DIR} (collection: {COLLECTION_NAME})")


if __name__ == "__main__":
    main()