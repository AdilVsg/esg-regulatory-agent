"""
Tests for src/agent/web_search.py.

The Tavily call is mocked: this sandbox has no network access to
Tavily's API, and even with a real key, live search results would
make this test non-deterministic anyway.
"""
from unittest.mock import MagicMock, patch

from src.agent.web_search import build_web_search


@patch("src.agent.web_search.TavilyClient")
def test_web_search_node_maps_tavily_results(mock_client_cls):
    mock_client_instance = MagicMock()
    mock_client_instance.search.return_value = {
        "results": [
            {
                "title": "Omnibus I overview",
                "url": "https://example.com/omnibus",
                "content": "The package simplifies CSRD reporting obligations.",
                "score": 0.9,
            },
            {
                "title": "CSRD amendment tracker",
                "url": "https://example.com/csrd-amendments",
                "content": "Adopted in 2025, effective from 2026.",
                "score": 0.8,
            },
        ]
    }
    mock_client_cls.return_value = mock_client_instance

    web_search_node = build_web_search()
    result = web_search_node({"question": "What changed in the Omnibus I package?"})

    assert result["web_results"] == [
        {
            "content": "The package simplifies CSRD reporting obligations.",
            "url": "https://example.com/omnibus",
            "title": "Omnibus I overview",
        },
        {
            "content": "Adopted in 2025, effective from 2026.",
            "url": "https://example.com/csrd-amendments",
            "title": "CSRD amendment tracker",
        },
    ]
    mock_client_instance.search.assert_called_once_with(
        "What changed in the Omnibus I package?", max_results=5
    )


@patch("src.agent.web_search.TavilyClient")
def test_web_search_node_handles_no_results(mock_client_cls):
    mock_client_instance = MagicMock()
    mock_client_instance.search.return_value = {"results": []}
    mock_client_cls.return_value = mock_client_instance

    web_search_node = build_web_search()
    result = web_search_node({"question": "Something obscure"})

    assert result["web_results"] == []