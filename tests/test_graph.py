"""
Integration test for src/agent/graph.py: verifies the three routes
dispatch to the right nodes and produce a well-formed final state.

Important gotcha this test works around: router.py and generate.py
both do `from google import genai`, so `src.agent.router.genai.Client`
and `src.agent.generate.genai.Client` are the exact same object in
memory, not two independent things. Patching both separately doesn't
give two independent mocks -- whichever patch is applied last wins for
both call sites, silently. So this file patches genai.Client exactly
once and uses a sequenced side_effect list on generate_content to
return the router's response first, then the generator's, matching
the real call order in the graph.
"""
from unittest.mock import MagicMock, patch

from src.agent.graph import build_graph
from src.agent.router import RouteDecision


def _router_response(route: str, doc_scope: str, reasoning: str) -> MagicMock:
    response = MagicMock()
    response.parsed = RouteDecision(route=route, doc_scope=doc_scope, reasoning=reasoning)
    return response


def _generate_response(text: str) -> MagicMock:
    response = MagicMock()
    response.text = text
    return response


@patch("src.agent.retrieve.retrieve")
@patch("google.genai.Client")
def test_rag_route_runs_retrieve_then_generate(mock_client_cls, mock_retrieve):
    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.side_effect = [
        _router_response("rag", "regulatory", "In the corpus."),
        _generate_response("Large undertakings [1]."),
    ]
    mock_client_cls.return_value = mock_client_instance

    mock_retrieve.return_value = [
        {"text": "CSRD applies to large undertakings.", "source": "01_csrd.pdf", "page": 3, "doc_type": "regulatory", "distance": 0.2}
    ]

    graph = build_graph()
    result = graph.invoke({"question": "Who does the CSRD apply to?"})

    assert result["route"] == "rag"
    assert result["answer"] == "Large undertakings [1]."
    assert result["sources"] == [{"label": "[1]", "detail": "01_csrd.pdf, p.3"}]
    assert mock_client_instance.models.generate_content.call_count == 2


@patch("google.genai.Client")
def test_refuse_route_skips_retrieve_and_generate(mock_client_cls):
    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.return_value = _router_response(
        "refuse", "both", "Off topic."
    )
    mock_client_cls.return_value = mock_client_instance

    graph = build_graph()
    result = graph.invoke({"question": "What is the capital of France?"})

    assert result["route"] == "refuse"
    assert "Off topic." in result["answer"]
    assert result["sources"] == []
    assert mock_client_instance.models.generate_content.call_count == 1


@patch("src.agent.web_search.TavilyClient")
@patch("google.genai.Client")
def test_web_search_route_runs_tavily_then_generate(mock_client_cls, mock_tavily_cls):
    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.side_effect = [
        _router_response("web_search", "both", "Asks about a 2026 update."),
        _generate_response("According to a live search, adopted in 2026 [1]."),
    ]
    mock_client_cls.return_value = mock_client_instance

    mock_tavily_instance = MagicMock()
    mock_tavily_instance.search.return_value = {
        "results": [
            {"title": "Delegated acts tracker", "url": "https://example.com/acts", "content": "Adopted in 2026...", "score": 0.9}
        ]
    }
    mock_tavily_cls.return_value = mock_tavily_instance

    graph = build_graph()
    result = graph.invoke({"question": "What changed in 2026?"})

    assert result["route"] == "web_search"
    assert result["answer"] == "According to a live search, adopted in 2026 [1]."
    assert result["sources"] == [{"label": "[1]", "detail": "Delegated acts tracker \u2014 https://example.com/acts"}]
    assert mock_client_instance.models.generate_content.call_count == 2