"""
Basic RAG chain: retrieve relevant chunks, then generate an answer with
Google Gemini (free tier, no credit card required), citing sources.
This is the week-1 baseline.

Next week this gets replaced by a LangGraph agent that also decides when
to route to regulatory-only, corporate-only, web search, or refusal.
For now, retrieval searches the whole corpus with no filter, which is
the known limitation the agent's routing will fix.
"""
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.retrieval.retriever import retrieve

load_dotenv()

client = genai.Client()  # reads GEMINI_API_KEY from the environment
MODEL = "gemini-3.6-flash"

SYSTEM_PROMPT = """You are an assistant answering questions about EU sustainability \
reporting regulation (CSRD, ESRS, EU Taxonomy) and how specific companies report \
against it, using only the excerpts provided below.

Rules:
- Answer only using the provided excerpts. If they don't contain the answer, say so \
explicitly rather than guessing.
- After every factual claim, cite the source in this format: (source, p.X).
- If excerpts from multiple documents disagree or address different aspects, say so.
- Keep the answer concise and factual. Do not give investment advice or opinions.
"""


def format_context(chunks):
    parts = []
    for c in chunks:
        parts.append(f"[{c['source']} p.{c['page']}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def answer(question, k=5, doc_type=None):
    chunks = retrieve(question, doc_type=doc_type, k=k)
    context = format_context(chunks)

    response = client.models.generate_content(
        model=MODEL,
        contents=f"Excerpts:\n\n{context}\n\nQuestion: {question}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
        ),
    )

    return {
        "question": question,
        "answer": response.text,
        "sources": [{"source": c["source"], "page": c["page"]} for c in chunks],
    }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What are the six environmental objectives of the EU Taxonomy Regulation?"
    result = answer(q)
    print(f"Q: {result['question']}\n")
    print(f"A: {result['answer']}\n")
    print("Sources used:")
    for s in result["sources"]:
        print(f"  - {s['source']} p.{s['page']}")