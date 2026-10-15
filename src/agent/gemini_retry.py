"""
Shared retry logic for Gemini API calls. Retries on HTTP 429 (quota
exceeded) and any 5xx server error (Google's infrastructure being
temporarily overloaded, not something wrong with the request). Does
NOT retry on other errors, like a 400 bad request or 401 auth failure,
since retrying won't fix those.

router.py used to carry its own copy of this logic. The duplication cost
exactly what duplication usually costs: a backoff bug sat in both copies,
invisible while the API answered, until a real saturation event exposed it
and it had to be fixed twice. The copy is gone and router.py imports from
here.
"""
from google import genai
from google.genai import errors as genai_errors
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)


def is_transient_error(exc: BaseException) -> bool:
    if isinstance(exc, genai_errors.ClientError) and getattr(exc, "code", None) == 429:
        return True
    return isinstance(exc, genai_errors.ServerError)


@retry(
    retry=retry_if_exception(is_transient_error),
    # multiplier=15 makes the backoff actually exponential: 15s, 30s, 60s,
    # 120s. The previous multiplier=1 produced 1s/2s/4s, all clamped up to
    # min=15, so every wait was a flat 15s and the whole policy gave up
    # after ~45s. A 503 "high demand" spike on a free-tier Flash-Lite model
    # routinely outlasts that.
    # wait_random adds jitter so a batch run doesn't retry in lockstep.
    wait=wait_exponential(multiplier=15, min=15, max=120) + wait_random(0, 5),
    stop=stop_after_attempt(6),
    reraise=True,
)
def generate_with_retry(client: genai.Client, **kwargs):
    return client.models.generate_content(**kwargs)