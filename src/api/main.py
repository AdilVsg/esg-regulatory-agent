"""
FastAPI service exposing the ESG regulatory agent over HTTP.

Builds the compiled LangGraph agent once at process startup (same
reasoning as the factory pattern already used in router.py/generate.py/
web_search.py: build expensive clients once, reuse across requests)
and exposes it through a single POST /query endpoint plus a GET
/health check for container orchestration / liveness probes.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent.graph import build_graph

_agent: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _agent["graph"] = build_graph()
    yield
    _agent.clear()


app = FastAPI(title="ESG Regulatory Agent", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to ask the ESG regulatory agent.")


class SourceOut(BaseModel):
    label: str
    detail: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    route: str | None = None
    doc_scope: str | None = None
    sources: list[SourceOut] = []


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    graph = _agent.get("graph")
    if graph is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        result = graph.invoke({"question": request.question})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}") from exc

    return QueryResponse(
        question=request.question,
        answer=result.get("answer", ""),
        route=result.get("route"),
        doc_scope=result.get("doc_scope"),
        sources=result.get("sources", []),
    )