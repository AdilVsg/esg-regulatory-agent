"""
Tests for src/api/main.py.

The agent graph is replaced by a mock, so these tests check the HTTP
contract only: the response shape on success, input validation, and that
a failure inside the agent never leaks its exception message to the client.
"""
import logging
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def graph():
    return MagicMock()


@pytest.fixture
def client(graph):
    # build_graph runs in the lifespan handler, which TestClient only
    # triggers when used as a context manager.
    with patch("src.api.main.build_graph", return_value=graph), TestClient(app) as c:
        yield c


def test_query_returns_answer_route_and_sources(client, graph):
    graph.invoke.return_value = {
        "answer": "Six objectives [1].",
        "route": "rag",
        "doc_scope": "regulatory",
        "sources": [{"label": "[1]", "detail": "03_taxonomy_regulation_2020_852.pdf, p.17"}],
    }

    response = client.post("/query", json={"question": "Six objectives?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Six objectives [1]."
    assert body["route"] == "rag"
    assert body["sources"][0]["label"] == "[1]"


def test_query_failure_does_not_leak_exception_details(client, graph, caplog):
    secret = "upstream failure at /app/src/agent/generate.py model=gemini-x key=abc123"
    graph.invoke.side_effect = RuntimeError(secret)

    with caplog.at_level(logging.ERROR, logger="src.api.main"):
        response = client.post("/query", json={"question": "anything"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Agent execution failed."}
    assert secret not in response.text
    # The details are not lost: they go to the server logs instead.
    assert secret in caplog.text


def test_empty_question_is_rejected_before_reaching_the_agent(client, graph):
    response = client.post("/query", json={"question": ""})

    assert response.status_code == 422
    graph.invoke.assert_not_called()
