"""
Run retrieval (no LLM involved, no API quota spent) against the
12-question test set and print the top result for each, so we can
check two things:

1. Is the top hit's source the one you'd actually expect? Q4 and Q5
   were pulling from Schneider Electric's report instead of the
   Taxonomy Regulation before doc_scope was added -- confirm they now
   correctly pull from 03_taxonomy_regulation_2020_852.pdf.
2. Is there a usable gap in the score between questions that should be
   answerable and ones that shouldn't (Q9, Q10)? The very first
   calibration run showed heavy overlap (0.571-0.769 across the
   board), so don't assume a clean threshold exists -- confirm or
   refute it with this rerun before wiring a threshold into the graph.

This script bypasses the router entirely (calling retrieve_node
directly, with no Gemini call), so doc_scope isn't available from a
real routing decision here. Instead it's derived from each question's
test-set category, which is exactly what doc_scope is meant to
capture: retrieval -> regulatory, cross_doc -> both. The other
categories (out_of_corpus, guardrail) never reach this node in the
real graph -- the router routes them to web_search or refuse before
retrieval happens -- but they're still run here with scope "both"
since this script probes retrieve_node directly, bypassing that.

Run from the repo root:
    py -3.12 -m eval.retrieve_calibration
"""
import yaml

from src.agent.retrieve import retrieve_node

CATEGORY_TO_SCOPE = {
    "retrieval": "regulatory",
    "cross_doc": "both",
}


def main():
    with open("eval/test_questions.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    for q in data["questions"]:
        doc_scope = CATEGORY_TO_SCOPE.get(q["category"], "both")
        result = retrieve_node({"question": q["question"], "doc_scope": doc_scope})
        docs = result["retrieved_docs"]
        top = docs[0] if docs else None

        print("=" * 80)
        print(f"{q['id']} [{q['category']}, doc_scope={doc_scope}] {q['question']}")
        if top:
            print(f"  top score: {top['score']:.3f}  ({top['source']} p.{top['page']}, {top['doc_type']})")
        else:
            print("  no results")
        print()


if __name__ == "__main__":
    main()