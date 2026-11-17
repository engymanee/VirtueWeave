import json
from types import SimpleNamespace

import pytest

from src.agent_demo import render_markdown as agent_markdown, run_demo
from src.compare_specialization import measure, render_markdown, run_comparison
from src.docs_section import replace_section
from src.llm import LLMUnavailable
from test_agent import GOOD, Script, reply, use


def test_replace_section_is_idempotent_and_leaves_the_rest_alone(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# Title\n\nkeep this\n", encoding="utf-8")
    replace_section(str(f), "x", "first")
    replace_section(str(f), "x", "second")
    text = f.read_text(encoding="utf-8")
    assert text.count("<!-- x:start -->") == 1 and "second" in text and "first" not in text
    assert text.startswith("# Title\n\nkeep this")
    replace_section(str(f), "other", "other section")
    assert "second" in f.read_text(encoding="utf-8") and "other section" in f.read_text(encoding="utf-8")


# ------------------------------------------------------------------- agent demo

DEMO = [{"topic": "loops", "grade": "3", "minutes": 20, "note": "unfair turns"}]


def test_agent_demo_saves_real_traces_and_the_readable_log(data_dir, tmp_path):
    runs = run_demo(DEMO, tool_call=Script(*GOOD), out_dir=str(tmp_path / "traces"),
                    doc_path=str(tmp_path / "ai.md"), data_dir=data_dir, trace_dir=str(tmp_path / "t"))
    saved = json.load(open(runs[0]["file"]))
    assert len(saved["steps"]) == 4 and saved["agent_top3"] == ["justice", "teamwork", "perseverance"]
    text = (tmp_path / "ai.md").read_text(encoding="utf-8")
    assert "agent workflow traces" in text and "`get_topic_fits`" in text
    assert "| 1 | `get_topic_fits` | see how each virtue fits |" in text
    assert "Agent's reasoning" in text and "agrees on" in text


def test_agent_demo_writes_nothing_if_the_agent_cannot_run(data_dir, tmp_path):
    def down(system, messages, tools):
        raise LLMUnavailable("no key")

    doc = tmp_path / "ai.md"
    with pytest.raises(SystemExit, match="Nothing was written"):
        run_demo(DEMO, tool_call=down, out_dir=str(tmp_path / "traces"), doc_path=str(doc),
                 data_dir=data_dir, trace_dir=str(tmp_path / "t"))
    assert not doc.exists()
    assert not list((tmp_path / "traces").glob("*.json"))


# ------------------------------------------------------------ specialization

SMALL = [
    {"topic": "loops", "grade": "1", "minutes": 15, "note": "they keep giving up"},
    {"topic": "events", "grade": "6", "minutes": 20, "note": ""},
]


def fake_model(system, user, schema):
    wanted = [v["virtue"] for v in json.loads(user)["virtues"]]
    few = "<example>" in system
    text = ("This fits well. It suits your class." if few else
            "A very long sentence with many words that goes on and on without stopping. Another. Third. Fourth.")
    return {"explanations": [{"virtue": v, "text": text} for v in wanted]}


def test_specialization_compares_the_same_picks_with_and_without_examples(data_dir, tmp_path):
    report = run_comparison(fake_model, SMALL[:1], data_dir=data_dir, trace_dir=str(tmp_path))
    assert report["requests"] == 1
    assert report["few_shot"]["guardrail_pass_rate"] == 1.0
    assert report["baseline"]["guardrail_pass_rate"] == 0.0     # four sentences fail the check
    assert report["few_shot"]["avg_sentences"] == 2.0 and report["baseline"]["avg_sentences"] == 4.0


def test_specialization_writes_nothing_if_the_model_is_down(data_dir, tmp_path):
    def down(system, user, schema):
        raise LLMUnavailable("no key")

    with pytest.raises(SystemExit, match="Nothing was written"):
        run_comparison(down, SMALL[:1], data_dir=data_dir, trace_dir=str(tmp_path))


def test_measures():
    pick = SimpleNamespace(virtue="honesty", title="I Found My Bug")
    m = measure({"honesty": 'Say it plainly. Try "I Found My Bug" today, they blame others.'},
                [pick], "some blame their partner")
    assert m["guardrail_pass"] is True and m["avg_sentences"] == 2.0
    assert m["names_activity"] == 1.0 and m["uses_note"] is True
    assert measure({"honesty": "One. Two."}, [pick], "")["uses_note"] is None


def test_specialization_markdown_has_the_table_and_an_example(data_dir, tmp_path):
    report = run_comparison(fake_model, SMALL, data_dir=data_dir, trace_dir=str(tmp_path))
    md = render_markdown(report)
    assert "| Measure | Baseline | Few-shot |" in md and "Passed the output guardrail" in md
    assert "- Baseline:" in md and "- Few-shot:" in md
