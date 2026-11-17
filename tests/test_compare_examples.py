import json

import pytest

from src.compare_examples import REQUESTS, run_comparison
from src.explainer import FEW_SHOT_EXAMPLES, MAX_CHARS, count_sentences
from src.llm import LLMUnavailable


def fake(system, user, schema):
    if "explanations" in schema["properties"]:
        wanted = [v["virtue"] for v in json.loads(user)["virtues"]]
        text = "Warm line one. Line two." + (" Extra." if "<example>" in system else "")
        return {"explanations": [{"virtue": v, "text": text} for v in wanted]}
    return {"topic": "loops", "age": 6, "minutes": 15, "description": "They keep giving up."}


def test_few_shot_examples_obey_the_guardrail_they_teach():
    for _, answer in FEW_SHOT_EXAMPLES:
        for item in answer["explanations"]:
            assert 2 <= count_sentences(item["text"]) <= 3
            assert len(item["text"]) <= MAX_CHARS


def test_comparison_runs_five_requests_both_ways_and_saves(tmp_path):
    report = run_comparison(fake, out_dir=str(tmp_path))
    assert len(report["rows"]) == len(REQUESTS) == 5
    for row in report["rows"]:
        assert set(row) >= {"with_examples", "without_examples", "picks"}
        assert row["without_examples"]["sentences"] == [2] * row["without_examples"]["virtues"]
        assert row["with_examples"]["sentences"] == [3] * row["with_examples"]["virtues"]
    assert report["summary"]["with_examples"]["guardrail_passes"] == 5
    assert json.load(open(report["path"]))["summary"] == report["summary"]


def test_comparison_stops_if_the_explainer_is_unavailable(tmp_path):
    def down(system, user, schema):
        if "explanations" in schema["properties"]:
            raise LLMUnavailable("no key")
        return fake(system, user, schema)
    with pytest.raises(SystemExit):
        run_comparison(down, out_dir=str(tmp_path))
