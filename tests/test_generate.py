"""
Tests for src/agent/generate.py.

The Gemini call is mocked, same reason as test_router.py: the suite
runs offline and in CI without an API key. These tests prove the
context-building and source-list wiring are correct, not that the
generated prose is good -- that is what the RAGAS evaluation measures.
"""
from unittest.mock import MagicMock, patch

from src.agent.generate import (
    _build_context_from_retrieved_docs,
    _build_context_from_web_results,
    build_generator,
)


def test_build_context_from_retrieved_docs_numbers_and_labels_sources():
    docs = [
        {"content": "chunk A text", "source": "01_csrd.pdf", "page": 12, "doc_type": "regulatory", "score": 0.7},
        {"content": "chunk B text", "source": "04_schneider.pdf", "page": 5, "doc_type": "corporate", "score": 0.6},
    ]

    context, sources = _build_context_from_retrieved_docs(docs)

    assert "[1]" in context and "chunk A text" in context
    assert "[2]" in context and "chunk B text" in context
    assert sources == [
        {"label": "[1]", "detail": "01_csrd.pdf, p.12"},
        {"label": "[2]", "detail": "04_schneider.pdf, p.5"},
    ]


def test_build_context_from_web_results_numbers_and_labels_sources():
    results = [
        {"content": "web snippet", "url": "https://example.com/a", "title": "Example A"},
    ]

    context, sources = _build_context_from_web_results(results)

    assert "[1]" in context and "web snippet" in context
    assert sources == [{"label": "[1]", "detail": "Example A \u2014 https://example.com/a"}]


@patch("src.agent.generate.genai.Client")
def test_generate_node_prefers_retrieved_docs_over_web_results(mock_client_cls):
    mock_response = MagicMock()
    mock_response.text = "The CSRD applies to large undertakings [1]."
    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.return_value = mock_response
    mock_client_cls.return_value = mock_client_instance

    generate_node = build_generator()
    state = {
        "question": "Who does the CSRD apply to?",
        "retrieved_docs": [
            {"content": "large undertakings...", "source": "01_csrd.pdf", "page": 3, "doc_type": "regulatory", "score": 0.8}
        ],
        "web_results": [
            {"content": "should not be used", "url": "https://example.com", "title": "Should not be used"}
        ],
    }

    result = generate_node(state)

    assert result["answer"] == "The CSRD applies to large undertakings [1]."
    assert result["sources"] == [{"label": "[1]", "detail": "01_csrd.pdf, p.3"}]
    _, call_kwargs = mock_client_instance.models.generate_content.call_args
    assert "large undertakings..." in call_kwargs["contents"]
    assert "should not be used" not in call_kwargs["contents"]


@patch("src.agent.generate.genai.Client")
def test_generate_node_falls_back_to_web_results(mock_client_cls):
    mock_response = MagicMock()
    mock_response.text = "According to a live search, the package was adopted in 2025 [1]."
    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.return_value = mock_response
    mock_client_cls.return_value = mock_client_instance

    generate_node = build_generator()
    state = {
        "question": "What did the Omnibus I package change?",
        "web_results": [
            {"content": "adopted in 2025...", "url": "https://example.com/omnibus", "title": "Omnibus I"}
        ],
    }

    result = generate_node(state)

    assert "live search" in result["answer"]
    assert result["sources"] == [{"label": "[1]", "detail": "Omnibus I \u2014 https://example.com/omnibus"}]


@patch("src.agent.generate.genai.Client")
def test_generate_node_handles_no_context(mock_client_cls):
    mock_response = MagicMock()
    mock_response.text = "No relevant information was found."
    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.return_value = mock_response
    mock_client_cls.return_value = mock_client_instance

    generate_node = build_generator()
    result = generate_node({"question": "Something with nothing retrieved"})

    assert result["sources"] == []