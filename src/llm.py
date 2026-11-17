"""
The only place in VirtueWeave that calls the model.

Every caller goes through call_json() (one request, one JSON object back) or
call_with_tools() (one request in a tool-use loop). If a call cannot produce a
usable answer for any reason (no credentials, network error, refusal, bad
JSON) it raises LLMUnavailable, and the caller decides on its fallback.

About temperature: this wrapper does NOT send temperature=0. On the current
API version (2026-10-21) a request that sets temperature, top_p or top_k gets
a 400 on every model, and the current SDK has no temperature argument. Output
is kept steady instead by a JSON schema (structured outputs), a short fixed
prompt and low effort. If you pin an older SDK and API version, this is the
single place to add temperature=0.
"""

import json
from typing import Any, Dict, List, Optional

MODEL = "claude-opus-5-5"   # swap for "claude-haiku-5-5" to cut cost on this small task
EFFORT = "low"
MAX_TOKENS = 1024
AGENT_MAX_TOKENS = 4096


class LLMUnavailable(Exception):
    """The model could not give a usable JSON answer. The message says why."""


def unavailable(*args: Any, **kwargs: Any) -> Any:
    """A stand-in for the model that always fails. Used for offline runs and in tests."""
    raise LLMUnavailable("the model is switched off")


def _send(client: Optional[Any], **request: Any) -> Any:
    """Sends one request and turns every failure into LLMUnavailable."""
    try:
        import anthropic
    except ImportError as error:
        raise LLMUnavailable("anthropic package is not installed") from error

    try:
        client = client or anthropic.Anthropic()
        response = client.messages.create(model=MODEL, **request)
    except anthropic.AnthropicError as error:  # auth, rate limit, network, server
        raise LLMUnavailable(f"{type(error).__name__}: {error}") from error
    except TypeError as error:
        # With no credentials at all the SDK raises a bare TypeError. Treat only
        # that one as "unavailable"; any other TypeError is a bug in our code.
        if "authentication method" not in str(error):
            raise
        raise LLMUnavailable("no API credentials found (set ANTHROPIC_API_KEY)") from error

    if response.stop_reason == "refusal":
        raise LLMUnavailable("the model declined the request")
    if response.stop_reason == "max_tokens":
        raise LLMUnavailable("the answer was cut off at max_tokens")
    return response


def call_json(
    system: str,
    user: str,
    schema: Dict[str, Any],
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Sends one request and returns the parsed JSON object.

    schema must be a JSON Schema for the answer ("additionalProperties": false
    and every key in "required"). client is for tests; by default the SDK
    resolves credentials from the environment.
    """
    response = _send(
        client,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={
            "effort": EFFORT,
            "format": {"type": "json_schema", "schema": schema},
        },
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise LLMUnavailable(f"the answer was not valid JSON: {error}") from error
    if not isinstance(data, dict):
        raise LLMUnavailable("the answer was JSON but not an object")
    return data


def call_with_tools(
    system: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    client: Optional[Any] = None,
) -> Any:
    """
    Sends one request of a tool-use loop and returns the response, whose content
    blocks include any tool_use blocks. The caller runs the tools, appends
    response.content and the tool results to messages, and calls again. Forced
    tool use is not available on this model, so the prompt asks for the tools.
    """
    return _send(
        client,
        max_tokens=AGENT_MAX_TOKENS,
        system=system,
        messages=messages,
        tools=tools,
        output_config={"effort": EFFORT},
    )
