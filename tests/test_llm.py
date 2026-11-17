from types import SimpleNamespace

import pytest

from src import llm
from src.llm import LLMUnavailable, call_json

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}},
          "required": ["a"], "additionalProperties": False}


class FakeClient:
    def __init__(self, text='{"a": 1}', stop_reason="end_turn"):
        self.calls = []
        self._response = SimpleNamespace(
            stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def _client_raising(error):
    def create(**kwargs):
        raise error
    return SimpleNamespace(messages=SimpleNamespace(create=create))


def test_returns_parsed_json_and_sends_the_schema():
    client = FakeClient()
    assert call_json("sys", "hi", SCHEMA, client=client) == {"a": 1}
    sent = client.calls[0]
    assert sent["model"] == llm.MODEL
    assert sent["output_config"]["format"] == {"type": "json_schema", "schema": SCHEMA}
    assert "temperature" not in sent  # rejected by the current API version


@pytest.mark.parametrize("client", [
    FakeClient(text="not json"),
    FakeClient(text="[1, 2]"),
    FakeClient(stop_reason="refusal"),
    FakeClient(stop_reason="max_tokens"),
])
def test_unusable_answers_raise_unavailable(client):
    with pytest.raises(LLMUnavailable):
        call_json("sys", "hi", SCHEMA, client=client)


def test_missing_credentials_typeerror_becomes_unavailable():
    error = TypeError("Could not resolve authentication method. Expected one of api_key")
    with pytest.raises(LLMUnavailable, match="no API credentials"):
        call_json("sys", "hi", SCHEMA, client=_client_raising(error))


def test_other_typeerror_is_not_swallowed():
    with pytest.raises(TypeError):
        call_json("sys", "hi", SCHEMA, client=_client_raising(TypeError("bad arg")))
