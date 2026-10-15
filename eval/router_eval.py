"""
Run the router against the 12-question test set and print its decision
for each one, so you can eyeball whether the classification matches
what you'd expect per category.

Requires a real GEMINI_API_KEY in the environment:

    $env:GEMINI_API_KEY = "..."
    python -m eval.router_eval

Expected mapping, as a starting point to eyeball against (adjust if
your test set's categories don't line up exactly this way):
    retrieval, cross_doc  -> "rag"
    out_of_corpus         -> "web_search" or "refuse", depending on the
                              question -- check each one individually
    guardrail             -> "refuse"

Results are checkpointed to eval/router_eval_results.jsonl as they
come in. If the script is interrupted, e.g. by a daily quota limit,
rerunning it skips questions that already have a result instead of
re-spending quota on them. Google's free tier RPD quota resets at
midnight Pacific Time, roughly 9am in Paris during summer. Delete
eval/router_eval_results.jsonl to force a full rerun, for example
after a prompt change.
"""
import json
import time
from pathlib import Path

import yaml

from src.agent.router import build_router

SECONDS_BETWEEN_REQUESTS = 13
RESULTS_PATH = Path("eval/router_eval_results.jsonl")


def load_existing_results() -> dict:
    if not RESULTS_PATH.exists():
        return {}
    results = {}
    with open(RESULTS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            results[record["id"]] = record
    return results


def print_record(record: dict) -> None:
    print("=" * 80)
    print(f"{record['id']} [{record['category']}] {record['question']}")
    print(f"  -> route: {record['route']}  |  doc_scope: {record['doc_scope']}")
    print(f"  -> reasoning: {record['reasoning']}")
    print()


def main():
    with open("eval/test_questions.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    total = len(data["questions"])
    existing = load_existing_results()

    for record in existing.values():
        print_record(record)

    remaining = [q for q in data["questions"] if q["id"] not in existing]

    if not remaining:
        print(f"All {total} questions already have results in {RESULTS_PATH}.")
        return

    print(f"{len(existing)}/{total} already done, {len(remaining)} left to run.\n")

    router_node = build_router()

    for i, q in enumerate(remaining):
        if i > 0:
            time.sleep(SECONDS_BETWEEN_REQUESTS)

        result = router_node({"question": q["question"]})
        record = {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "route": result["route"],
            "doc_scope": result["doc_scope"],
            "reasoning": result["route_reasoning"],
        }

        with open(RESULTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        print_record(record)

    print(f"Done. {len(load_existing_results())}/{total} questions have results in {RESULTS_PATH}.")


if __name__ == "__main__":
    main()