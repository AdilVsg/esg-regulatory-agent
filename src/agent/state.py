"""
Shared state schema for the ESG regulatory agent LangGraph graph.

This TypedDict is the single object passed between every node. Each
node reads the fields it needs and returns a partial dict of updates,
which LangGraph merges into the running state. total=False means every
field is optional at any given point in the graph's execution, since
most fields are only populated after a specific node has run.
"""
from typing import Literal, TypedDict

from langchain_core.messages import BaseMessage


class RetrievedDoc(TypedDict):
    content: str
    source: str
    page: int
    doc_type: Literal["regulatory", "corporate"]
    score: float


class WebResult(TypedDict):
    content: str
    url: str
    title: str


class Source(TypedDict):
    label: str
    detail: str


class ESGAgentState(TypedDict, total=False):
    # Input
    question: str
    chat_history: list[BaseMessage]

    # Router output
    route: Literal["rag", "web_search", "refuse"]
    doc_scope: Literal["regulatory", "corporate", "both"]
    route_reasoning: str

    # Retrieval output
    retrieved_docs: list[RetrievedDoc]
    retrieval_score: float  # best chunk similarity score, 0-1

    # Web search output
    web_results: list[WebResult]

    # Final output
    answer: str
    sources: list[Source]