"""
Assembles the week-2 LangGraph agent:

    Router -> "rag"        -> Retrieve    -> Generate -> END
           -> "web_search"  -> Web search -> Generate -> END
           -> "refuse"      -> Refuse                 -> END
"""
from langgraph.graph import END, StateGraph

from src.agent.generate import build_generator
from src.agent.retrieve import retrieve_node
from src.agent.router import build_router
from src.agent.state import ESGAgentState
from src.agent.web_search import build_web_search


def _refuse_node(state: ESGAgentState) -> dict:
    reasoning = state.get("route_reasoning", "this is outside the scope of the ESG regulatory corpus.")
    return {
        "answer": f"I can't help with that: {reasoning}",
        "sources": [],
    }


def _route_from_router(state: ESGAgentState) -> str:
    return state["route"]


def build_graph():
    """
    Builds and compiles the agent graph. Call this once per process
    (it builds the Gemini and Tavily clients internally), then call
    .invoke({"question": ...}) as many times as needed.
    """
    graph = StateGraph(ESGAgentState)

    graph.add_node("router", build_router())
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("web_search", build_web_search())
    graph.add_node("generate", build_generator())
    graph.add_node("refuse", _refuse_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        _route_from_router,
        {
            "rag": "retrieve",
            "web_search": "web_search",
            "refuse": "refuse",
        },
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("refuse", END)

    return graph.compile()