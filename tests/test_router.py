"""
Tests for src/agent/router.py.

Two things are checked here, deliberately kept separate:

1. Schema validity: does RouteDecision reject an invalid route or
   doc_scope? This runs for real, no mocking needed, since it's pure
   Pydantic.

2. Node wiring: does router_node correctly turn a RouteDecision into
   the state update dict, including doc_scope? The Gemini call itself
   is mocked, so the suite runs offline and in CI without an API key.
   This test proves the plumbing is correct, not that the
   classification is accurate. Real classification quality must be
   checked by running eval/router_eval.py against the 12-question test
   set with a real GEMINI_API_KEY.
"""
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.agent.router import RouteDecision, build_router

def test_route_decision_accepts_valid_route_and_doc_scope():
    decision = RouteDecision(route="rag", doc_scope="regulatory", reasoning="In corpus.")
    assert decision.route == "rag"
    assert decision.doc_scope == "regulatory"


def test_route_decision_rejects_invalid_route():
    with pytest.raises(ValidationError):
        RouteDecision(route="not_a_real_route", doc_scope="both", reasoning="Should fail.")


def test_route_decision_rejects_invalid_doc_scope():
    with pytest.raises(ValidationError):
        RouteDecision(route="rag", doc_scope="not_a_real_scope", reasoning="Should fail.")


@patch("src.agent.router.genai.Client")
def test_router_node_maps_decision_to_state_update(mock_client_cls):
    fake_decision = RouteDecision(
        route="web_search",
        doc_scope="both",
        reasoning="Asks for the latest amendment, post-dates the corpus.",
    )
    mock_response = MagicMock()
    mock_response.parsed = fake_decision
    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.return_value = mock_response
    mock_client_cls.return_value = mock_client_instance

    router_node = build_router()
    result = router_node({"question": "What is the latest amendment to the CSRD?"})

    assert result == {
        "route": "web_search",
        "route_reasoning": "Asks for the latest amendment, post-dates the corpus.",
        "doc_scope": "both",
    }


@patch("src.agent.router.genai.Client")
def test_router_node_passes_question_and_system_prompt(mock_client_cls):
    fake_decision = RouteDecision(route="refuse", doc_scope="both", reasoning="Off topic.")
    mock_response = MagicMock()
    mock_response.parsed = fake_decision
    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.return_value = mock_response
    mock_client_cls.return_value = mock_client_instance

    router_node = build_router()
    router_node({"question": "What is the best pizza topping?"})

    _, call_kwargs = mock_client_instance.models.generate_content.call_args
    assert call_kwargs["contents"] == "What is the best pizza topping?"
    assert call_kwargs["config"].system_instruction is not None
    assert call_kwargs["config"].response_schema is RouteDecision


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))