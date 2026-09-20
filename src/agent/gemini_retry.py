"""
Shared retry logic for Gemini API calls. Retries on HTTP 429 (quota
exceeded) and any 5xx server error (Google's infrastructure being
temporarily overloaded, not something wrong with the request). Does
NOT retry on other errors, like a 400 bad request or 401 auth failure,
since retrying won't fix those.

router.py has its own local copy of this same logic rather than
importing it from here, since it was written first and is already
tested -- not worth the churn of touching a stable, tested file just
to deduplicate ~15 lines. Worth unifying in a later cleanup pass if
this project keeps growing.
"""
from google import genai
from google.genai import errors as genai_errors
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


def is_transient_error(exc: BaseException) -> bool:
    if isinstance(exc, genai_errors.ClientError) and getattr(exc, "code", None) == 429:
        return True
    return isinstance(exc, genai_errors.ServerError)


@retry(
    retry=retry_if_exception(is_transient_error),
    wait=wait_exponential(multiplier=1, min=15, max=60),
    stop=stop_after_attempt(4),
    reraise=True,
)
def generate_with_retry(client: genai.Client, **kwargs):
    return client.models.generate_content(**kwargs)