"""
Generate node: produces the final answer, grounded in whichever
context the graph populated -- retrieved_docs from the corpus
(route="rag") or web_results from the web search fallback
(route="web_search"). retrieved_docs takes priority if both are
somehow present, since the corpus is the higher-trust source.

The [1], [2]... source list is built deterministically from the
context here, not asked of the LLM: citations are a transformation of
already-known data, not something that needs the model's judgment, and
free text is more reliable for prose than trying to get an LLM to
also emit structured citation objects correctly.
"""
from google import genai
from google.genai import types

from src.agent.gemini_retry import generate_with_retry
from src.agent.state import ESGAgentState, Source

import os

DEFAULT_GENERATE_MODEL = os.environ.get("GENERATE_MODEL", "gemini-3.5-flash-lite")
GENERATE_SYSTEM_PROMPT = """You are an ESG regulatory assistant. Answer the question using ONLY the numbered context passages provided below it.

Rules:
- Cite the passage(s) you draw on inline using their bracketed number, e.g. [1], [2]. Cite every claim that comes from the context.
- If the context doesn't fully answer the question, say plainly what's missing rather than filling the gap from general knowledge.
- Be precise and concise. No generic ESG commentary that isn't grounded in the context.
- If the context passages are web search results rather than the internal document corpus, say so in the answer, so the reader knows this wasn't verified against the curated corpus."""


def _build_context_from_retrieved_docs(docs: list) -> tuple[str, list[Source]]:
    context_parts = []
    sources: list[Source] = []
    for i, doc in enumerate(docs, start=1):
        label = f"[{i}]"
        context_parts.append(f"{label} (source: {doc['source']}, p.{doc['page']})\n{doc['content']}")
        sources.append({"label": label, "detail": f"{doc['source']}, p.{doc['page']}"})
    return "\n\n".join(context_parts), sources


def _build_context_from_web_results(results: list) -> tuple[str, list[Source]]:
    context_parts = []
    sources: list[Source] = []
    for i, r in enumerate(results, start=1):
        label = f"[{i}]"
        context_parts.append(f"{label} (source: {r['title']}, {r['url']})\n{r['content']}")
        sources.append({"label": label, "detail": f"{r['title']} \u2014 {r['url']}"})
    return "\n\n".join(context_parts), sources


def build_generator(model: str = DEFAULT_GENERATE_MODEL):
    """
    Factory returning a LangGraph-compatible generate node. Kept as a
    factory for the same reason as build_router: build the client once,
    reuse it across calls.
    """
    client = genai.Client()

    def generate_node(state: ESGAgentState) -> dict:
        retrieved_docs = state.get("retrieved_docs")
        web_results = state.get("web_results")

        if retrieved_docs:
            context, sources = _build_context_from_retrieved_docs(retrieved_docs)
        elif web_results:
            context, sources = _build_context_from_web_results(web_results)
        else:
            context, sources = "(no context retrieved)", []

        prompt = f"Question: {state['question']}\n\nContext:\n{context}"

        response = generate_with_retry(
            client,
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=GENERATE_SYSTEM_PROMPT,
                temperature=0.2,
            ),
        )

        return {
            "answer": response.text,
            "sources": sources,
        }

    return generate_node