"""
Tests for the shared Gemini retry policy.

This predicate used to be duplicated in router.py under the name
_is_rate_limit_error, and its test lived in test_router.py. The duplication
hid a backoff bug that had to be fixed in two places, so the logic was
unified here and the test moved with it.
"""
from google.genai import errors as genai_errors

from src.agent.gemini_retry import is_transient_error


def test_is_transient_error_matches_429_and_5xx():
    rate_limited = genai_errors.ClientError(429, {"error": {"message": "quota"}})
    bad_request = genai_errors.ClientError(400, {"error": {"message": "bad request"}})
    server_overloaded = genai_errors.ServerError(503, {"error": {"message": "unavailable"}})

    assert is_transient_error(rate_limited) is True
    assert is_transient_error(server_overloaded) is True
    assert is_transient_error(bad_request) is False
    assert is_transient_error(ValueError("unrelated")) is False