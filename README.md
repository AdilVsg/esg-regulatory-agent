# ESG Regulatory Agent

An agentic RAG assistant that answers questions on EU sustainability reporting regulation (CSRD, ESRS, EU Taxonomy) by combining retrieval over a curated regulatory corpus with a live web search fallback for information published after the corpus cutoff.

## Why this project

Static RAG systems answer well on the documents they were built on, but regulatory text changes. This project demonstrates an agent that recognizes the boundary of its own knowledge base and routes accordingly, rather than confidently answering from a stale corpus.

## Architecture

```
User question
     |
     v
 [LangGraph router]
     |
     +--> in-corpus question --> [vector retriever] --> [LLM answer + citations]
     |
     +--> out-of-corpus / recent --> [web search tool] --> [LLM answer + sources]
     |
     +--> out-of-scope --> [refusal]
     v
 Answer + source attribution
```

- **Ingestion**: PDF/HTML loading, chunking, embedding into a persistent vector store
- **Retrieval**: similarity search over the regulatory + corporate report corpus
- **Agent**: LangGraph state machine deciding retrieval vs. web search vs. refusal
- **Evaluation**: RAGAS metrics (faithfulness, answer relevancy, context precision/recall) against a hand-built test set
- **Serving**: FastAPI backend, Streamlit frontend, containerized and deployed on Google Cloud Run

## Tech stack

| Layer | Tool |
|---|---|
| Orchestration | LangGraph |
| Vector store | Chroma DB |
| Embeddings | sentence-transformers |
| LLM | OpenAI / Anthropic API |
| Evaluation | RAGAS |
| API | FastAPI |
| UI | Streamlit |
| Containerization | Docker |
| Deployment | Google Cloud Run, Artifact Registry, Secret Manager |
| CI/CD | GitHub Actions |

## Project status

- [x] Scope definition and corpus selection
- [x] Test question set (v1)
- [x] Corpus downloaded and answer locations verified by hand
- [ ] Ingestion pipeline (chunking + embedding)
- [ ] Baseline RAG (retrieval + citations)
- [ ] LangGraph routing (retrieval vs. web search vs. refusal)
- [ ] RAGAS evaluation harness
- [ ] FastAPI service
- [ ] Dockerization
- [ ] Cloud Run deployment + CI/CD
- [ ] Streamlit UI

## Corpus

See [`data/raw/SOURCES.md`](data/raw/SOURCES.md) for the list of source documents and download instructions.

## Evaluation

See [`eval/test_questions.yaml`](eval/test_questions.yaml) for the test question set used to evaluate retrieval quality and routing behavior.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in API keys
```

## License

MIT
