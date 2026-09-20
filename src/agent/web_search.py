"""
Web search node: used when the router determines the question needs
current information beyond the static corpus (e.g. recent regulatory
amendments, post-corpus events). Uses Tavily, the search API most
commonly paired with LangGraph agents; free tier covers 1,000
requests/month, no credit card.
"""
from tavily import TavilyClient

from src.agent.state import ESGAgentState, WebResult

MAX_RESULTS = 5


def build_web_search():
    """
    Factory returning a LangGraph-compatible web search node. Kept as
    a factory for the same reason as build_router/build_generator:
    build the client once, reuse it across calls. TavilyClient() reads
    TAVILY_API_KEY from the environment automatically.
    """
    client = TavilyClient()

    def web_search_node(state: ESGAgentState) -> dict:
        response = client.search(state["question"], max_results=MAX_RESULTS)

        web_results: list[WebResult] = [
            {
                "content": r["content"],
                "url": r["url"],
                "title": r["title"],
            }
            for r in response.get("results", [])
        ]

        return {"web_results": web_results}

    return web_search_node