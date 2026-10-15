"""
Router node: classifies the incoming question into one of three routes
before any retrieval happens.

- "rag": the question is answerable from the curated corpus (CSRD,
  ESRS 1 and ESRS E1, EU Taxonomy Regulation, Schneider Electric and
  BNP Paribas sustainability reports).
- "web_search": the question concerns EU sustainability regulation or
  one of the two companies, but explicitly asks for something recent
  or post-dating the corpus (e.g. "the latest amendment", "as of
  2026", "has this changed since").
- "refuse": the question is unrelated to EU sustainability reporting
  regulation or to the two named companies' disclosures.

Uses Google Gemini (free tier, no credit card) via the google-genai
SDK, the same client already used in src/agent/basic_rag.py for week
1. Structured output is enforced with response_schema set to a
Pydantic model, so the SDK returns a parsed, validated object instead
of free text to parse ourselves.
"""
from typing import Literal
import os

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from src.agent.state import ESGAgentState
from src.agent.gemini_retry import generate_with_retry

# Router classification is a lightweight task and doesn't need full Flash
# reasoning. Overridable via env var because model availability has shifted
# four times on this project: gpt-4o-mini, then gemini-2.5-flash unavailable,
# then gemini-3.6-flash's 20 RPD cap, then gemini-3.1-flash-lite saturating
# for a whole evening in September 2026. The default below is the model the
# published evaluation was run on. Check what's available on your account:
#   c = genai.Client(); [print(m.name) for m in c.models.list()]
DEFAULT_ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "gemini-3.5-flash-lite")
ROUTER_SYSTEM_PROMPT = """You are the routing component of an ESG regulatory assistant.

Your knowledge base is a fixed, offline snapshot of exactly five documents:
1. The CSRD directive (Directive (EU) 2022/2464) -- full text, including \
scope and size thresholds.
2. ESRS 1 (General Requirements) and ESRS E1 (Climate Change), from the \
ESRS Set 1 delegated regulation -- including materiality, phase-in \
reliefs, and GHG Scope 1/2/3 disclosure requirements.
3. The EU Taxonomy Regulation (Regulation (EU) 2020/852) -- including \
its six environmental objectives and the Do No Significant Harm (DNSH) \
criteria.
4. Schneider Electric's 2024 sustainability report.
5. BNP Paribas's 2025 sustainability report.

The full text of these documents is already stored and searchable. A \
question about what a regulation SAYS, DEFINES, or REQUIRES is \
answerable from the knowledge base even though the topic is \
"regulatory" -- you do not need the internet to read a document you \
already have. Route these to "rag".

Route to "web_search" ONLY when the question explicitly asks about \
something the corpus cannot contain: a change, amendment, or update \
that happened AFTER these documents were downloaded, or the \
"latest"/"current"/"most recent" state of a rule (e.g. "has this \
threshold changed", "what's the latest amendment").

Route to "refuse" if the question is unrelated to these five documents \
(other companies, general ESG chit-chat, legal advice, unrelated \
topics).

Examples:
- "What are the six environmental objectives of the EU Taxonomy \
Regulation?" -> rag (this is the literal content of a corpus document)
- "What does the DNSH principle require?" -> rag (defined in the \
Taxonomy Regulation, which is in the corpus)
- "What are the Scope 1/2/3 disclosure requirements under ESRS E1?" \
-> rag (defined in ESRS E1, which is in the corpus)
- "Which size thresholds trigger CSRD scope?" -> rag (defined in the \
CSRD directive, which is in the corpus)
- "Has the CSRD size threshold changed since the directive was \
published?" -> web_search (asks about a change over time, not the \
original text)
- "What is Tesla's Scope 3 emissions figure?" -> refuse (Tesla is not \
in the corpus)

Be conservative in the other direction too: don't default to \
"web_search" just because a topic sounds like it could change \
eventually. Only route there when the question itself signals \
recency or change.

Additionally classify doc_scope, which only matters when route is \
"rag":
- "regulatory": the question asks what a regulation itself defines, \
states, or requires (e.g. "what are the six environmental \
objectives", "what does DNSH require"). Use this even if a company's \
report happens to restate similar content -- the regulation is the \
authoritative source for what a regulation says, not a company's \
paraphrase of it.
- "corporate": the question is purely about what a company disclosed, \
with no comparison to a regulatory requirement.
- "both": the question compares a company's disclosure against a \
regulatory requirement.
When route is "web_search" or "refuse", set doc_scope to "both" \
(it goes unused).

Examples:
- "What are the six environmental objectives of the EU Taxonomy \
Regulation?" -> rag, doc_scope=regulatory (asks what the regulation \
itself defines, not how any company applies it)
- "How does Schneider Electric's reported Scope 3 emissions \
methodology compare with the requirements set out in ESRS E1?" -> \
rag, doc_scope=both (explicitly compares a company's disclosure \
against a regulatory requirement)"""


class RouteDecision(BaseModel):
    route: Literal["rag", "web_search", "refuse"] = Field(
        description="The selected route for this question."
    )
    doc_scope: Literal["regulatory", "corporate", "both"] = Field(
        description=(
            "Only meaningful when route is 'rag'. Which part of the "
            "knowledge base actually answers this question. Default "
            "to 'both' when route is not 'rag'."
        )
    )
    reasoning: str = Field(
        description="One short sentence explaining the classification."
    )


def build_router(model: str = DEFAULT_ROUTER_MODEL):
    """
    Factory returning a LangGraph-compatible router node.

    Kept as a factory (rather than a bare function) so the client is
    built once and reused across every invocation of the node, instead
    of being reconstructed on every question. genai.Client() reads
    GEMINI_API_KEY from the environment automatically, same as
    basic_rag.py.
    """
    client = genai.Client()

    def router_node(state: ESGAgentState) -> dict:
        response = generate_with_retry(
            client,
            model=model,
            contents=state["question"],
            config=types.GenerateContentConfig(
                system_instruction=ROUTER_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=RouteDecision,
            ),
        )
        decision: RouteDecision = response.parsed
        return {
            "route": decision.route,
            "route_reasoning": decision.reasoning,
            "doc_scope": decision.doc_scope,
        }

    return router_node