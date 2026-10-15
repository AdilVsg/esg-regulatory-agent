"""
Run the full agent graph over the test set and checkpoint every result
to eval/rag_outputs.jsonl. This is the expensive half of the RAGAS
harness: it spends Gemini quota (router + generate = 2 calls per
question). Scoring is done separately by eval/run_ragas.py, which
reads this file, so a failed scoring run never costs generation quota.

All 12 questions are generated, not just the 8 RAG ones. The 4 extra
calls give a complete record of the agent's behaviour on the whole
set, including the refusals, which is worth having in the repo.

Gold answers are deliberately NOT written here. run_ragas.py joins them
from eval/test_questions.yaml at scoring time, so editing a reference
answer does not require regenerating anything.

Usage, from the repo root:

    $env:GEMINI_API_KEY = "..."
    $env:TAVILY_API_KEY = "..."
    py -3.12 -m eval.generate_answers

Interrupted runs resume: questions already present in the output file
are skipped. Delete eval/rag_outputs.jsonl to force a full rerun.
"""
import json
import time
import traceback
from pathlib import Path
import os

import yaml

from src.agent.graph import build_graph

# Router + generate = 2 Gemini calls per question. router_eval.py uses 13s
# for a single call; 20s here keeps the same margin under the free-tier RPM.
SECONDS_BETWEEN_QUESTIONS = 20

QUESTIONS_PATH = Path("eval/test_questions.yaml")
OUTPUTS_PATH = Path("eval/rag_outputs.jsonl")


def extract_contexts(result: dict) -> list[str]:
    """
    Pull the raw chunk text out of the graph result.

    RetrievedDoc declares "content" in state.py while retriever.py returns
    "text", so accept either rather than guess which one survives the
    conversion in retrieve.py. Web results are captured too, for the record,
    even though RAGAS will not score the web route.
    """
    docs = result.get("retrieved_docs") or result.get("web_results") or []
    contexts = []
    for doc in docs:
        text = doc.get("content") or doc.get("text")
        if text is None:
            raise KeyError(
                "No 'content' or 'text' key in a retrieved doc. "
                f"Keys present: {sorted(doc)}. Fix extract_contexts()."
            )
        contexts.append(text)
    return contexts


def load_done_ids() -> set[str]:
    if not OUTPUTS_PATH.exists():
        return set()
    done = set()
    with open(OUTPUTS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["id"])
    return done


def main() -> None:
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = yaml.safe_load(f)["questions"]

    done = load_done_ids()
    remaining = [q for q in questions if q["id"] not in done]

    if not remaining:
        print(f"All {len(questions)} questions already in {OUTPUTS_PATH}.")
        return

    print(f"{len(done)}/{len(questions)} done, {len(remaining)} to run.")
    print(f"Roughly {len(remaining) * SECONDS_BETWEEN_QUESTIONS // 60} min of pacing.\n")

    graph = build_graph()
    failures = []

    for i, q in enumerate(remaining):
        if i > 0:
            time.sleep(SECONDS_BETWEEN_QUESTIONS)

        print(f"[{q['id']}] {q['question'][:70]}...")

        try:
            result = graph.invoke({"question": q["question"]})
        except Exception:
            # One bad question must not throw away the questions already
            # generated in this run: they are on disk, and the id is skipped
            # on the next pass anyway. Report and move on.
            print(f"  FAILED\n{traceback.format_exc()}")
            failures.append(q["id"])
            continue

        record = {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            # The models are part of what is being measured: RAGAS scores
            # describe this configuration, not whatever default the repo
            # happens to carry. Recorded per question so a run that was
            # interrupted and resumed under a different model is detectable
            # instead of silently mixing two systems in one score.
            # The models are part of what is being measured: RAGAS scores describe
            # this configuration, not whatever default the repo happens to carry.
            "router_model": os.environ.get("ROUTER_MODEL", "gemini-3.5-flash-lite"),
            "generate_model": os.environ.get("GENERATE_MODEL", "gemini-3.5-flash-lite"),
            "top_k": None,  # filled below
            "route": result.get("route"),
            "doc_scope": result.get("doc_scope"),
            "route_reasoning": result.get("route_reasoning"),
            "retrieval_score": result.get("retrieval_score"),
            "answer": result.get("answer"),
            "retrieved_contexts": extract_contexts(result),
            "sources": result.get("sources", []),
        }

        with open(OUTPUTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        n_ctx = len(record["retrieved_contexts"])
        score = record["retrieval_score"]
        score_str = f"{score:.3f}" if isinstance(score, float) else "n/a"
        print(f"  route={record['route']}  chunks={n_ctx}  score={score_str}")

    total_done = len(load_done_ids())
    print(f"\n{total_done}/{len(questions)} in {OUTPUTS_PATH}.")
    if failures:
        print(f"Failed, rerun to retry: {', '.join(failures)}")


if __name__ == "__main__":
    main()