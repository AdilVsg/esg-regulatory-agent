"""
Tests for src/agent/retrieve.py.

The Chroma call itself is mocked: this sandbox has no network access
to Hugging Face to download the BAAI/bge-small-en-v1.5 model weights,
and no access to your local Chroma database either. These tests prove
the doc_scope routing and score conversion are correct, not that real
retrieval quality is good on your corpus -- check that empirically
with eval/retrieve_calibration.py.
"""
from unittest.mock import patch

from src.agent.retrieve import _distance_to_score, retrieve_node


def test_distance_to_score_is_bounded_and_monotonic():
    assert _distance_to_score(0.0) == 1.0
    assert 0.0 < _distance_to_score(1.0) < 1.0
    assert _distance_to_score(1.0) < _distance_to_score(0.5)
    assert _distance_to_score(10.0) > 0.0


@patch("src.agent.retrieve.retrieve")
def test_regulatory_scope_only_queries_regulatory_pool(mock_retrieve):
    mock_retrieve.return_value = [
        {
            "text": "taxonomy chunk",
            "source": "03_taxonomy_regulation_2020_852.pdf",
            "page": 4,
            "doc_type": "regulatory",
            "distance": 0.3,
        }
    ]

    result = retrieve_node({"question": "six environmental objectives", "doc_scope": "regulatory"})

    mock_retrieve.assert_called_once_with(
        "six environmental objectives", doc_type="regulatory", k=5
    )
    assert result["retrieved_docs"][0]["source"] == "03_taxonomy_regulation_2020_852.pdf"


@patch("src.agent.retrieve.retrieve")
def test_corporate_scope_only_queries_corporate_pool(mock_retrieve):
    mock_retrieve.return_value = [
        {
            "text": "schneider chunk",
            "source": "04_schneider_electric_urd_2024.pdf",
            "page": 12,
            "doc_type": "corporate",
            "distance": 0.4,
        }
    ]

    result = retrieve_node({"question": "Schneider's own emissions figures", "doc_scope": "corporate"})

    mock_retrieve.assert_called_once_with(
        "Schneider's own emissions figures", doc_type="corporate", k=5
    )
    assert result["retrieved_docs"][0]["source"] == "04_schneider_electric_urd_2024.pdf"


@patch("src.agent.retrieve.retrieve")
def test_both_scope_queries_and_merges_both_pools(mock_retrieve):
    def fake_retrieve(query, doc_type=None, k=5):
        if doc_type == "regulatory":
            return [
                {
                    "text": "regulatory chunk",
                    "source": "02_esrs_set1_2023_2772.pdf",
                    "page": 81,
                    "doc_type": "regulatory",
                    "distance": 0.3,
                }
            ]
        if doc_type == "corporate":
            return [
                {
                    "text": "corporate chunk",
                    "source": "04_schneider_electric_urd_2024.pdf",
                    "page": 127,
                    "doc_type": "corporate",
                    "distance": 0.1,
                }
            ]
        raise AssertionError(f"unexpected doc_type={doc_type}")

    mock_retrieve.side_effect = fake_retrieve

    result = retrieve_node({"question": "compare Schneider's methodology to ESRS E1", "doc_scope": "both"})

    assert mock_retrieve.call_count == 2
    # Lower distance wins regardless of pool when doc_scope is "both".
    assert result["retrieved_docs"][0]["source"] == "04_schneider_electric_urd_2024.pdf"
    assert result["retrieved_docs"][1]["source"] == "02_esrs_set1_2023_2772.pdf"


@patch("src.agent.retrieve.retrieve")
def test_missing_doc_scope_defaults_to_both(mock_retrieve):
    mock_retrieve.return_value = []

    retrieve_node({"question": "no doc_scope key at all"})

    assert mock_retrieve.call_count == 2
    called_doc_types = {call.kwargs["doc_type"] for call in mock_retrieve.call_args_list}
    assert called_doc_types == {"regulatory", "corporate"}


@patch("src.agent.retrieve.retrieve")
def test_retrieve_node_handles_empty_results(mock_retrieve):
    mock_retrieve.return_value = []

    result = retrieve_node({"question": "Something with no matches", "doc_scope": "regulatory"})

    assert result["retrieved_docs"] == []
    assert result["retrieval_score"] == 0.0