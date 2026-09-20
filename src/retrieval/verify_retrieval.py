"""
Sanity-check the vector store: run the retrieval and cross_doc questions
from eval/test_questions.yaml and print the top-3 chunks for each, so you
can compare them by eye against the expected_source noted in that file.

This is a manual QA step, not the RAGAS evaluation (that comes later,
once the LangGraph agent exists).
"""
import yaml
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = "data/processed/chroma"
COLLECTION_NAME = "esg_corpus"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 5


def main():
    with open("eval/test_questions.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    checkable = [q for q in data["questions"] if q["category"] in ("retrieval", "cross_doc")]

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)

    for q in checkable:
        print("=" * 80)
        print(f"{q['id']} [{q['category']}] {q['question']}")
        print(f"Expected: {q.get('expected_source', '(not set)')}")
        print("-" * 80)

        if q["category"] == "retrieval":
            filters = [("regulatory only", {"doc_type": "regulatory"})]
        else:
            filters = [("regulatory only", {"doc_type": "regulatory"}), ("corporate only", {"doc_type": "corporate"})]

        for label, where in filters:
            print(f"  -- {label} --")
            results = collection.query(query_texts=[q["question"]], n_results=TOP_K, where=where)
            for rank, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), start=1):
                preview = doc[:180].replace("\n", " ")
                print(f"  #{rank} {meta['source']} p.{meta['page']}: {preview}...")
        print()


if __name__ == "__main__":
    main()