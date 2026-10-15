"""
Retrieve node: queries the Chroma collection built in week 1 for the
top-k most relevant chunks to the question, and computes a confidence
score used later by the graph to decide whether to fall back to web
search.

Reuses src/retrieval/retriever.py from week 1 rather than duplicating
the Chroma connection logic. Which doc_type pool(s) get searched is
driven by doc_scope, set by the router: merging both pools and
sorting by raw distance turned out not to be enough, because a
company's own restatement of a regulatory concept can score closer to
the question than the regulation's actual (more formally worded) text
-- Q4 and Q5 in the 12-question test set hit exactly this. The router
already classifies this correctly as part of the same structured
output used for routing, at no extra API cost, so Retrieve trusts it
instead of re-deriving scope from distance.
"""
from src.agent.state import ESGAgentState, RetrievedDoc
from src.retrieval.retriever import retrieve

TOP_K = 10


def _distance_to_score(distance: float) -> float:
    """
    Converts a Chroma distance (0 = identical, larger = less similar,
    no fixed upper bound) into a score in (0, 1] where higher means
    more confident. Deliberately doesn't assume a specific distance
    metric or upper bound (L2 vs cosine), since that depends on how
    the collection was created -- this only relies on distance >= 0
    and being monotonically decreasing in similarity, which holds
    either way.
    """
    return 1.0 / (1.0 + distance)


def retrieve_node(state: ESGAgentState) -> dict:
    question = state["question"]
    doc_scope = state.get("doc_scope", "both")

    if doc_scope == "regulatory":
        chunks = retrieve(question, doc_type="regulatory", k=TOP_K)
    elif doc_scope == "corporate":
        chunks = retrieve(question, doc_type="corporate", k=TOP_K)
    else:
        regulatory_chunks = retrieve(question, doc_type="regulatory", k=TOP_K)
        corporate_chunks = retrieve(question, doc_type="corporate", k=TOP_K)
        # doc_scope "both" means the question compares a company disclosure with a
        # regulatory requirement, so both sides must survive into the context.
        # Merging the pools and truncating to TOP_K did not guarantee that:
        # whichever document scored better could take every slot. Observed on
        # 2026-09-23, Q7 retrieved five Schneider chunks and zero ESRS ones, and
        # the agent correctly refused to answer a comparison it had no basis for.
        half = TOP_K // 2
        chunks = sorted(
            regulatory_chunks[:half] + corporate_chunks[:TOP_K - half],
            key=lambda c: c["distance"],
        )

    retrieved_docs: list[RetrievedDoc] = [
        {
            "content": c["text"],
            "source": c["source"],
            "page": c["page"],
            "doc_type": c["doc_type"],
            "score": _distance_to_score(c["distance"]),
        }
        for c in chunks
    ]

    best_score = retrieved_docs[0]["score"] if retrieved_docs else 0.0

    return {
        "retrieved_docs": retrieved_docs,
        "retrieval_score": best_score,
    }