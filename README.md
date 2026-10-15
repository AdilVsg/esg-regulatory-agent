# ESG Regulatory Agent
![CI](https://github.com/AdilVsg/esg-regulatory-agent/actions/workflows/ci.yml/badge.svg)

An agentic RAG assistant that answers questions on EU sustainability reporting
regulation (CSRD, ESRS, EU Taxonomy) by combining retrieval over a curated
regulatory corpus with a live web search fallback for information published
after the corpus cutoff.

## Why this project

Static RAG systems answer well on the documents they were built on, but
regulatory text changes. This project demonstrates an agent that recognizes the
boundary of its own knowledge base and routes accordingly, rather than
confidently answering from a stale corpus.

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

The router emits a structured Pydantic decision with two fields: the `route`
above, and a `doc_scope` of `regulatory`, `corporate` or `both` that selects
which half of the corpus is searched. Questions comparing a company disclosure
against a regulatory requirement get `both`, and the retrieval budget is split
evenly between the two pools so neither side can be crowded out.

- **Ingestion**: PDF loading, chunking (1000 chars, 150 overlap), embedding into
  a persistent Chroma store with a `doc_type` metadata split
- **Retrieval**: similarity search, `TOP_K = 10`, filtered by `doc_scope`
- **Agent**: LangGraph state machine over a shared `ESGAgentState` TypedDict
- **Evaluation**: routing accuracy and RAGAS answer-quality metrics against a
  hand-built 12-question set
- **Serving**: FastAPI, containerized and deployed on Google Cloud Run

## Corpus

Five documents, split into a regulatory and a corporate pool. They are not
committed to this repository; see
[`data/raw/SOURCES.md`](data/raw/SOURCES.md) for the source list and download
instructions.

| File | Document | Pool |
|---|---|---|
| `01_csrd_directive_2022_2464.pdf` | CSRD, Directive (EU) 2022/2464 | regulatory |
| `02_esrs_set1_2023_2772.pdf` | ESRS Set 1, Regulation (EU) 2023/2772 (ESRS 1, ESRS E1) | regulatory |
| `03_taxonomy_regulation_2020_852.pdf` | EU Taxonomy, Regulation (EU) 2020/852 | regulatory |
| `04_schneider_electric_urd_2024.pdf` | Schneider Electric Universal Registration Document 2024 | corporate |
| `05_bnp_paribas_urd_2025.pdf` | BNP Paribas Universal Registration Document 2025 | corporate |

## Tech stack

| Layer | Tool |
|---|---|
| Orchestration | LangGraph |
| Vector store | Chroma DB |
| Embeddings | `BAAI/bge-small-en-v1.5` (sentence-transformers, local) |
| LLM | Google Gemini (`google-genai`) |
| Web search | Tavily |
| Evaluation | RAGAS |
| API | FastAPI |
| Containerization | Docker |
| Deployment | Google Cloud Run, Artifact Registry, Secret Manager |

Model names are overridable without touching code, via the `ROUTER_MODEL` and
`GENERATE_MODEL` environment variables. This is not decoration: Google retired
or saturated four different Gemini models over the life of this project,
including one that stopped being available to new accounts mid-development.

## API

Two endpoints:

```bash
curl https://<your-cloud-run-url>/health

curl -X POST https://<your-cloud-run-url>/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the six environmental objectives of the EU Taxonomy Regulation?"}'
```

The response carries the answer, the route the agent took, and the sources it
cited:

```json
{
  "question": "...",
  "answer": "Based on the provided documents, the six environmental objectives ...",
  "route": "rag",
  "doc_scope": "regulatory",
  "sources": [
    {"label": "[1]", "detail": "03_taxonomy_regulation_2020_852.pdf, p.17"}
  ]
}
```

## Evaluation

Two things are measured separately, because they fail in different ways and a
single aggregate would hide both: whether the router sends a question down the
right path, and whether the RAG path produces a grounded answer.

### Routing

`eval/router_eval.py` runs the router node alone against the 12-question set in
`eval/test_questions.yaml` and records its decision for each. Results are
checkpointed to `eval/router_eval_results.jsonl`, and the expected route per
category was checked by hand against that file.

| Category | Questions | Expected route | Correct |
|---|---|---|---|
| retrieval | Q1-Q6 | `rag` | 6/6 |
| cross_doc | Q7-Q8 | `rag` | 2/2 |
| out_of_corpus | Q9-Q10 | `web_search` | 2/2 |
| guardrail | Q11-Q12 | `refuse` | 2/2 |

12/12. The two refusal questions retrieve nothing at all, which is the intended
behaviour: the guardrail fires before any vector search happens.

### Answer quality (RAGAS)

`eval/run_ragas.py` scores the eight RAG questions. Generation and scoring are
two separate scripts: `eval/generate_answers.py` calls the agent once and writes
`eval/rag_outputs.jsonl`, and the scorer reads that file. A failed scoring run
therefore never costs generation quota a second time, and the same answers can
be rescored after a metric is adjusted, which keeps comparisons valid.

Run of 2026-09-24. Agent on `gemini-3.5-flash-lite` for both router and
generator, `TOP_K = 10`, judge `gemini-3.5-flash-lite`, embeddings running
locally so that scoring spends no embedding quota.

| Metric | Score |
|---|---|
| faithfulness | 0.902 |
| answer_relevancy | 0.783 |

| Question | faithfulness | answer_relevancy |
|---|---|---|
| Q1 CSRD scope | 1.000 | 0.806 |
| Q2 double materiality | 0.917 | 0.878 |
| Q3 ESRS E1 Scopes 1/2/3 | 1.000 | 0.876 |
| Q4 Taxonomy objectives | 1.000 | 1.000 |
| Q5 DNSH | 0.800 | 0.862 |
| Q6 phase-in reliefs | 1.000 | 0.858 |
| Q7 Schneider vs ESRS E1 | 0.500 | 0.000 |
| Q8 BNP Green Asset Ratio | 1.000 | 0.983 |

### What the evaluation found

The first run scored `answer_relevancy` at 0.459, with Q1, Q5 and Q7 at exactly
zero. RAGAS assigns zero to answers it classifies as non-committal, and these
three were declining to answer for lack of context. Inspecting the retrieved
chunks showed why, and produced two fixes:

- **`TOP_K` raised from 5 to 10.** Questions whose answer is spread across
  distant parts of one document were missing it. For Q1 the decisive chunks
  (Article 19a, Article 5(2)) ranked 7th and 8th; for Q5, Article 17 ranked 8th.
- **Pool balancing on cross-document questions.** When `doc_scope` is `both`,
  `retrieve_node` queried the regulatory and corporate pools separately, merged
  them and truncated to `TOP_K`. Nothing guaranteed both sides survived: Q7
  retrieved five Schneider chunks and zero ESRS ones, making the comparison it
  asked for structurally unanswerable. Half the budget is now reserved per pool,
  and `tests/test_retrieve.py` pins that behaviour.

`answer_relevancy` went from 0.459 to 0.783 as a result.

Q7 still scores zero. Pool balancing works, five chunks now come from each
document, but single-query embedding retrieval does not locate the right
passages for a question mixing two subjects: the ESRS chunks came from pages 81,
112, 113 and 116 rather than the E1-6 section on pages 91 to 93. Fixing this
needs query decomposition, which is a real architectural change and is not
implemented.

### Limitations

- **Eight questions is a very small sample.** These scores indicate a tendency,
  not a benchmark. A single question moves a mean by more than 0.1.
- **The context metrics are not reported.** `context_precision` and
  `context_recall` require the judge to emit a structured verdict per retrieved
  chunk. The only judge models with a free-tier daily quota large enough for a
  RAGAS run are Flash-Lite, and they produce verdicts that contradict the other
  metrics: `context_precision` at 0.000 on questions where `faithfulness` is
  1.000 and `context_recall` is non-zero, which cannot both be true. The scores
  are recorded in `eval/ragas_results.json` but are not interpretable as
  measurements. Flash-class judges have a 20 requests-per-day free-tier quota,
  against the 60 to 100 calls a run needs.
- **Q1 cannot be fully answered from the corpus.** The CSRD cross-references
  Article 3(4) of Directive 2013/34/EU for the numeric size thresholds, and that
  directive is not in the corpus. The agent says so rather than filling the gap
  from parametric knowledge, which is the intended behaviour.
- **PDF text extraction is imperfect.** Official Journal PDFs encode kerning in
  a way that inserts spaces inside words (`Ar ticle`, `en vironmentally`). The
  embeddings tolerate it, but it is present in the indexed chunks.
- **No conversational memory.** `chat_history` exists in the state schema but is
  unused; every question is answered independently.
- **`retrieval_score` is computed but unused.** The best chunk similarity is
  recorded in the state and ranges from 0.61 to 0.77 across the test set, which
  is enough spread to drive a confidence threshold. No threshold is wired in.

## Project status

- [x] Scope definition and corpus selection
- [x] Test question set (v1), answer locations verified by hand
- [x] Ingestion pipeline (chunking + embedding)
- [x] Baseline RAG (retrieval + citations)
- [x] LangGraph routing (retrieval vs. web search vs. refusal)
- [x] RAGAS evaluation harness
- [x] FastAPI service
- [x] Dockerization
- [x] Cloud Run deployment
- [ ] CI/CD (GitHub Actions)
- [ ] Query decomposition for cross-document questions

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # fill in GEMINI_API_KEY and TAVILY_API_KEY
```

Build the vector store, then ask a question:

```bash
python -m src.ingestion.chunk_and_embed
python ask.py "What are the six environmental objectives of the EU Taxonomy Regulation?"
```

Run the API locally:

```bash
uvicorn src.api.main:app --reload --port 8080
```

### Evaluation and tests

Evaluation dependencies are kept out of the production image, which serves the
agent rather than evaluating it:

```bash
pip install -r requirements-dev.txt
pytest -q
python -m eval.router_eval          # routing decisions, ~12 Gemini calls
python -m eval.generate_answers     # agent answers -> eval/rag_outputs.jsonl
python -m eval.run_ragas            # scores -> eval/ragas_results.json
```

Both eval scripts checkpoint as they go and skip questions that already have a
result, so an interrupted run resumes without re-spending quota.

## License

MIT
