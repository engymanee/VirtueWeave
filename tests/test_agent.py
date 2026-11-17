import json
from types import SimpleNamespace

import pytest

from src.agent import TOOLS, run_agent
from src.catalog import load_catalog
from src.llm import LLMUnavailable
from src.pipeline import Request, run_pipeline
from src.retrieval import VirtueRetriever


def say(text):
    return SimpleNamespace(type="text", text=text)


def use(name, call_id="t", **args):
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=args)


def reply(*blocks):
    return SimpleNamespace(content=list(blocks), stop_reason="tool_use")


class Script:
    """A fake model: returns the prepared replies in order and records what it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, system, messages, tools):
        self.calls.append([dict(m) for m in messages])
        return self.replies.pop(0)


GOOD = [
    reply(say("First I will look at the topic."),
          use("get_topic_fits", "a", reason="see how each virtue fits")),
    reply(use("search_class_notes", "b", query="unfair turns", reason="read the note")),
    reply(use("get_activity", "c", virtue="justice", reason="check the time")),
    reply(use("submit_recommendation", "d", virtues=["justice", "teamwork", "perseverance"],
              reasoning="Justice matches the note; the others fit the topic.")),
]


@pytest.fixture
def setup(data_dir):
    catalog = load_catalog(data_dir)
    request = Request("loops", "3", 20, "they argue over turns", "3-5")
    return catalog, request, VirtueRetriever.from_catalog(catalog)


def test_the_agent_uses_tools_then_submits_three_virtues(setup):
    catalog, request, retriever = setup
    run = run_agent(request, catalog, retriever, Script(*GOOD))
    assert run.error is None
    assert run.virtues == ["justice", "teamwork", "perseverance"]
    assert [s["tool"] for s in run.steps] == [
        "get_topic_fits", "search_class_notes", "get_activity", "submit_recommendation"]
    assert [s["step"] for s in run.steps] == [1, 2, 3, 4]


def test_every_step_records_the_agents_reason(setup):
    catalog, request, retriever = setup
    run = run_agent(request, catalog, retriever, Script(*GOOD))
    assert run.steps[0]["said"] == "First I will look at the topic."
    assert run.steps[0]["reason"] == "see how each virtue fits"
    assert run.steps[2]["input"] == {"virtue": "justice"}
    assert "Justice matches the note" in run.steps[3]["reason"]
    assert run.reasoning.startswith("Justice matches")


def test_tool_results_come_from_the_catalog(setup):
    catalog, request, retriever = setup
    script = Script(*GOOD)
    run = run_agent(request, catalog, retriever, script)
    results = {}
    for messages in script.calls[1:]:
        last = messages[-1]["content"][0]
        results[last["tool_use_id"]] = json.loads(last["content"])
    assert set(results["a"]["fits"]) == set(catalog.virtues)
    assert results["a"]["fits"]["perseverance"]["fit"] == 1.0
    assert results["b"]["matches"] and results["b"]["matches"][0]["virtue"] == "justice"
    assert results["c"]["minutes"] == 15 and results["c"]["fits_time"] is True
    assert run.steps[3]["result"] == '{"accepted": true}'   # the last result is never sent back


def test_the_model_is_sent_the_request_and_the_tools(setup):
    catalog, request, retriever = setup
    script = Script(*GOOD)
    run_agent(request, catalog, retriever, script)
    task = json.loads(script.calls[0][0]["content"])
    assert task == {"topic": "loops", "grade": "3", "grade_band": "3-5", "minutes_available": 20,
                    "teacher_note": "they argue over turns"}
    assert {t["name"] for t in TOOLS} == {
        "get_topic_fits", "search_class_notes", "get_activity", "submit_recommendation"}


@pytest.mark.parametrize("bad", [
    ["justice", "teamwork"],                    # too few
    ["justice", "justice", "teamwork"],         # duplicate
    ["justice", "teamwork", "patience"],        # not a virtue
])
def test_an_invalid_submission_is_corrected_and_the_agent_can_retry(setup, bad):
    catalog, request, retriever = setup
    script = Script(
        reply(use("submit_recommendation", "x", virtues=bad, reasoning="first try")),
        reply(use("submit_recommendation", "y", virtues=["honesty", "teamwork", "justice"], reasoning="fixed")),
    )
    run = run_agent(request, catalog, retriever, script)
    assert run.virtues == ["honesty", "teamwork", "justice"]
    assert '"accepted": false' in run.steps[0]["result"].lower()


def test_an_empty_reasoning_is_rejected(setup):
    catalog, request, retriever = setup
    script = Script(
        reply(use("submit_recommendation", "x", virtues=["justice", "teamwork", "honesty"], reasoning=" ")),
        reply(use("submit_recommendation", "y", virtues=["justice", "teamwork", "honesty"], reasoning="ok")),
    )
    assert run_agent(request, catalog, retriever, script).virtues == ["justice", "teamwork", "honesty"]


def test_repeated_invalid_submissions_give_up(setup):
    catalog, request, retriever = setup
    bad = lambda i: reply(use("submit_recommendation", f"x{i}", virtues=["a"], reasoning="r"))
    run = run_agent(request, catalog, retriever, Script(bad(1), bad(2), bad(3)))
    assert run.virtues is None and "invalid" in run.error


def test_stopping_without_a_submission_is_an_error(setup):
    catalog, request, retriever = setup
    run = run_agent(request, catalog, retriever, Script(reply(say("I think justice."))))
    assert run.virtues is None and "without submitting" in run.error


def test_too_many_steps_is_an_error(setup):
    catalog, request, retriever = setup
    loop = [reply(use("get_topic_fits", f"t{i}", reason="again")) for i in range(5)]
    run = run_agent(request, catalog, retriever, Script(*loop), max_steps=3)
    assert run.virtues is None and "within 3 steps" in run.error and len(run.steps) == 3


def test_unknown_tools_and_bad_arguments_return_errors_not_crashes(setup):
    catalog, request, retriever = setup
    script = Script(
        reply(use("delete_everything", "a", reason="x")),
        reply(use("get_activity", "b", virtue="patience", reason="x")),
        *GOOD,
    )
    run = run_agent(request, catalog, retriever, script)
    assert "unknown tool" in run.steps[0]["result"]
    assert "unknown virtue" in run.steps[1]["result"]
    assert run.virtues == ["justice", "teamwork", "perseverance"]


def test_no_note_means_the_search_tool_finds_nothing(setup):
    catalog, _, _ = setup
    request = Request("loops", "3", 20, "", "3-5")
    script = Script(
        reply(use("search_class_notes", "a", query="", reason="check")),
        reply(use("submit_recommendation", "b", virtues=["justice", "teamwork", "honesty"], reasoning="r")),
    )
    run = run_agent(request, catalog, None, script)
    assert "no note" in run.steps[0]["result"]


def test_a_model_failure_propagates_so_the_caller_can_fall_back(setup):
    catalog, request, retriever = setup

    def down(system, messages, tools):
        raise LLMUnavailable("down")

    with pytest.raises(LLMUnavailable):
        run_agent(request, catalog, retriever, down)


# ------------------------------------------------------------- in the pipeline

def explain_down(system, user, schema):
    raise LLMUnavailable("explain off")


def test_pipeline_agent_mode_uses_the_agents_choice_and_records_its_trace(data_dir, tmp_path):
    result = run_pipeline("loops", "3", 20, "they argue over turns", data_dir=data_dir,
                          trace_dir=str(tmp_path / "t"), model_call=explain_down,
                          agent=True, tool_call=Script(*GOOD))
    assert [p.virtue for p in result.picks] == ["justice", "teamwork", "perseverance"]
    assert result.agent["used"] and result.agent["overlap"] in (2, 3)
    assert any("chosen by the agent" in r for r in result.picks[0].reasons)
    trace = json.load(open(result.trace_path))
    agent_step = next(s for s in trace["steps"] if s["step"] == "agent")
    assert agent_step["status"] == "ok" and len(agent_step["steps"]) == 4
    assert agent_step["scorer_top3"] and agent_step["agent_top3"] == ["justice", "teamwork", "perseverance"]
    assert "Chosen by the agent" in __import__("src.pipeline", fromlist=["x"]).format_result(result)


def test_pipeline_agent_mode_falls_back_to_the_scorer_when_the_model_is_down(data_dir, tmp_path):
    def down(system, messages, tools):
        raise LLMUnavailable("no key")

    result = run_pipeline("loops", "3", 20, "", data_dir=data_dir, trace_dir=str(tmp_path / "t"),
                          model_call=explain_down, agent=True, tool_call=down)
    assert result.ok and [p.virtue for p in result.picks] == ["perseverance", "teamwork", "justice"]
    assert result.agent == {"used": False, "fallback_reason": "model unavailable: no key"}


def test_pipeline_agent_mode_falls_back_when_the_agent_never_submits(data_dir, tmp_path):
    result = run_pipeline("loops", "3", 20, "", data_dir=data_dir, trace_dir=str(tmp_path / "t"),
                          model_call=explain_down, agent=True,
                          tool_call=Script(reply(say("Done."))))
    assert result.agent["used"] is False and "without submitting" in result.agent["fallback_reason"]
    assert [p.virtue for p in result.picks] == ["perseverance", "teamwork", "justice"]


def test_scorer_mode_is_unchanged_and_has_no_agent_step(data_dir, tmp_path):
    result = run_pipeline("loops", "3", 20, "", data_dir=data_dir, trace_dir=str(tmp_path / "t"),
                          model_call=explain_down)
    assert result.agent is None
    steps = [s["step"] for s in json.load(open(result.trace_path))["steps"]]
    assert "agent" not in steps
