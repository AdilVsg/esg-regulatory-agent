"""
Ask a single question to the full agent graph and print the
whole trace: route, doc_scope, retrieval score, sources, and the
final answer. Uses real Gemini calls (router + generate), so it needs
GEMINI_API_KEY set.

Usage:
    python ask.py "What are the CSRD size thresholds?"

With no argument, it prompts interactively instead.
"""
import sys

from src.agent.graph import build_graph


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Question: ")

    graph = build_graph()
    result = graph.invoke({"question": question})

    print("=" * 80)
    print(f"Question: {question}")
    print(f"Route: {result.get('route')}  |  doc_scope: {result.get('doc_scope')}")
    print(f"Router reasoning: {result.get('route_reasoning')}")
    if result.get("retrieval_score") is not None:
        print(f"Retrieval score: {result['retrieval_score']:.3f}")
    print()
    print("Answer:")
    print(result.get("answer"))
    print()
    if result.get("sources"):
        print("Sources:")
        for s in result["sources"]:
            print(f"  {s['label']} {s['detail']}")


if __name__ == "__main__":
    main()