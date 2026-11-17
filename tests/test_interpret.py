from types import SimpleNamespace

import pytest

from src import llm
from src.interpret import (
    Interpretation, interpret, keyword_interpret, load_topics,
)
from src.llm import LLMUnavailable, call_json

SENTENCE = "Teaching loops to my 6-year-olds tomorrow, 15 minutes, they keep giving up."
TOPICS = [
    "algorithms", "sequencing", "loops", "events", "conditionals", "variables",
    "debugging", "testing", "decomposition", "patterns", "pair programming",
    "sharing a project",
]
GOOD = {"topic": "loops", "age": 6, "minutes": 15, "description": "They keep giving up."}


def model_returning(data):
    return lambda system, user, schema: data


def model_failing(message="no key"):
    def call(system, user, schema):
        raise LLMUnavailable(message)
    return call


def test_catalog_has_twelve_topics():
    assert len(load_topics("data/practices.csv")) == 12


def test_valid_model_answer_is_used():
    result = interpret(SENTENCE, TOPICS, model_returning(GOOD))
    assert result == Interpretation("loops", 6, 15, "They keep giving up.", "model")


def test_topic_is_normalised_before_the_check():
    result = interpret(SENTENCE, TOPICS, model_returning({**GOOD, "topic": " Loops "}))
    assert result.source == "model" and result.topic == "loops"


@pytest.mark.parametrize("bad", [
    {"topic": "recursion"},            # not in the catalog
    {"topic": None},                   # no topic
    {"age": "six"},                    # not a number
    {"age": 0},
    {"minutes": -5},
    {"minutes": True},
    {"description": "  "},
])
def test_answer_failing_a_check_falls_back(bad):
    result = interpret(SENTENCE, TOPICS, model_returning({**GOOD, **bad}))
    assert result.source == "keyword"
    assert "rejected" in result.fallback_reason
    assert (result.topic, result.age, result.minutes) == ("loops", 6, 15)


def test_null_age_and_minutes_are_allowed():
    result = interpret(SENTENCE, TOPICS, model_returning({**GOOD, "age": None, "minutes": None}))
    assert result.source == "model" and result.age is None and result.minutes is None


def test_model_unavailable_falls_back_with_reason():
    result = interpret(SENTENCE, TOPICS, model_failing("no API key"))
    assert result.source == "keyword"
    assert "no API key" in result.fallback_reason


def test_empty_sentence_is_an_error():
    with pytest.raises(ValueError):
        interpret("   ", TOPICS, model_returning(GOOD))


def test_keyword_parser_reads_the_example_sentence():
    result = keyword_interpret(SENTENCE, TOPICS)
    assert (result.topic, result.age, result.minutes) == ("loops", 6, 15)
    assert "15" not in result.description and "year" not in result.description
    assert "giving up" in result.description


@pytest.mark.parametrize("text,topic", [
    ("My class is learning to debug and find bugs", "debugging"),
    ("We are doing if-then cards today", "conditionals"),
    ("Kids will work in pair programming", "pair programming"),
    ("They will be sharing their project with parents", "sharing a project"),
    ("Practising a sequence of blocks", "sequencing"),
])
def test_keyword_parser_finds_topics(text, topic):
    assert keyword_interpret(text, TOPICS).topic == topic


def test_keyword_parser_returns_none_when_nothing_matches():
    assert keyword_interpret("The weather is nice", TOPICS).topic is None


def test_keyword_parser_reads_age_in_other_forms():
    assert keyword_interpret("kids aged 7, 10 min", TOPICS).age == 7
    assert keyword_interpret("a 5 year old class", TOPICS).age == 5
    assert keyword_interpret("no number here", TOPICS).age is None


# ---------------------------------------------------------------- wrapper

class FakeClient:
    def __init__(self, text='{"a": 1}', stop_reason="end_turn"):
        self.calls = []
        self._response = SimpleNamespace(
            stop_reason=stop_reason,
            content=[SimpleNamespace(type="text", text=text)],
        )
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}},
          "required": ["a"], "additionalProperties": False}


def _client_raising(error):
    def create(**kwargs):
        raise error
    return SimpleNamespace(messages=SimpleNamespace(create=create))


def test_missing_credentials_typeerror_becomes_unavailable():
    error = TypeError("Could not resolve authentication method. Expected one of api_key")
    with pytest.raises(LLMUnavailable, match="no API credentials"):
        call_json("sys", "hi", SCHEMA, client=_client_raising(error))


def test_other_typeerror_is_not_swallowed():
    with pytest.raises(TypeError):
        call_json("sys", "hi", SCHEMA, client=_client_raising(TypeError("bad arg")))


def test_wrapper_returns_parsed_json_and_sends_schema():
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
def test_wrapper_raises_unavailable_on_unusable_answers(client):
    with pytest.raises(LLMUnavailable):
        call_json("sys", "hi", SCHEMA, client=client)
