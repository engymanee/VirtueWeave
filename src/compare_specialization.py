"""
Measures what the few-shot examples change: the same picks are explained twice
by the real model, once with the examples in the prompt (specialized) and once
without (baseline). Both prompts carry the same rules; only the examples differ.

    python3 -m src.compare_specialization

Needs an API key (8 requests x 2 calls = 16 short calls). Results go to
assets/specialization_results.json and into model_card.md. It stops without
writing anything if the model cannot be reached.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List

from src.catalog import load_catalog
from src.docs_section import replace_section
from src.explainer import count_sentences, explain, guard_output
from src.llm import LLMUnavailable, call_json, unavailable
from src.pipeline import run_pipeline
from src.retrieval import _tokens

REQUESTS = [
    {"topic": "loops", "grade": "1", "minutes": 15, "note": "they keep giving up"},
    {"topic": "pair programming", "grade": "7", "minutes": 20, "note": "they argue over who gets the keyboard"},
    {"topic": "conditionals", "grade": "2", "minutes": 10, "note": "a few kids tease classmates who get it wrong"},
    {"topic": "sharing a project", "grade": "4", "minutes": 20, "note": "the kids never say thank you"},
    {"topic": "debugging", "grade": "5", "minutes": 15, "note": "some blame their partner when it breaks"},
    {"topic": "variables", "grade": "10", "minutes": 30, "note": ""},
    {"topic": "patterns", "grade": "K", "minutes": 10, "note": ""},
    {"topic": "algorithms", "grade": "11", "minutes": 30, "note": "they rush and skip steps"},
]


def measure(explanations: Dict[str, str], picks: List[Any], note: str) -> Dict[str, Any]:
    texts = {p.virtue: explanations.get(p.virtue, "") for p in picks}
    sentences = [count_sentences(t) for t in texts.values()]
    words = [len(t.split()) for t in texts.values()]
    note_words = set(_tokens(note))
    mentions = bool(note_words) and any(note_words & set(_tokens(t)) for t in texts.values())
    return {
        "guardrail_pass": guard_output(explanations, picks) == "",
        "avg_sentences": sum(sentences) / len(sentences),
        "words_per_sentence": sum(words) / max(1, sum(sentences)),
        "names_activity": sum(f'"{p.title}"' in texts[p.virtue] for p in picks) / len(picks),
        "uses_note": mentions if note else None,
    }


def run_comparison(
    model_call: Callable[..., Dict] = call_json,
    requests: List[Dict[str, Any]] = REQUESTS,
    data_dir: str = "data",
    trace_dir: str = "traces",
) -> Dict[str, Any]:
    guides = load_catalog(data_dir).guides()
    rows = []
    for req in requests:
        base = run_pipeline(req["topic"], req["grade"], req["minutes"], req["note"],
                            data_dir=data_dir, trace_dir=trace_dir, model_call=unavailable)
        row = {"request": req}
        for label, with_examples in (("baseline", False), ("few_shot", True)):
            try:
                text = explain(base.request, base.picks, guides, model_call, with_examples)
            except LLMUnavailable as error:
                raise SystemExit(f"Stopped: the model is unavailable ({error}). Nothing was written.")
            except ValueError:
                text = {}
            row[label] = {"text": text, **measure(text, base.picks, req["note"])}
        rows.append(row)

    def summary(label: str) -> Dict[str, float]:
        group = [r[label] for r in rows]
        with_note = [g for g in group if g["uses_note"] is not None]
        return {
            "guardrail_pass_rate": sum(g["guardrail_pass"] for g in group) / len(group),
            "avg_sentences": sum(g["avg_sentences"] for g in group) / len(group),
            "words_per_sentence": sum(g["words_per_sentence"] for g in group) / len(group),
            "names_activity_rate": sum(g["names_activity"] for g in group) / len(group),
            "uses_note_rate": (sum(g["uses_note"] for g in with_note) / len(with_note)) if with_note else None,
        }

    return {"requests": len(rows), "baseline": summary("baseline"), "few_shot": summary("few_shot"), "rows": rows}


def _fmt(value: Any, percent: bool) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.0f}%" if percent else f"{value:.1f}"


def render_markdown(report: Dict[str, Any]) -> str:
    b, f = report["baseline"], report["few_shot"]
    metrics = [
        ("Passed the output guardrail", "guardrail_pass_rate", True),
        ("Sentences per virtue (target 2-3)", "avg_sentences", False),
        ("Words per sentence (lower is plainer)", "words_per_sentence", False),
        ("Names the activity in quotes", "names_activity_rate", True),
        ("Connects to the teacher's note", "uses_note_rate", True),
    ]
    out = [
        "## Specialization: baseline vs few-shot examples",
        "",
        f"The explainer was run on {report['requests']} requests with the real model, once with two "
        "few-shot examples in the prompt (specialized) and once without (baseline). Both prompts have "
        "the same rules, and each request's picks are identical across the two runs, so the only "
        "difference is the examples. Generated by `python3 -m src.compare_specialization`; full output "
        "in `assets/specialization_results.json`.",
        "",
        "| Measure | Baseline | Few-shot |",
        "|---|---|---|",
    ]
    for label, key, percent in metrics:
        out.append(f"| {label} | {_fmt(b[key], percent)} | {_fmt(f[key], percent)} |")
    first = report["rows"][0]
    r = first["request"]
    virtue = next(iter(first["few_shot"]["text"] or first["baseline"]["text"]), None)
    if virtue:
        out += ["", f"**Example** ({r['topic']}, grade {r['grade']}, {r['minutes']} min, "
                f"note \"{r['note']}\"), virtue: {virtue}", "",
                f"- Baseline: {first['baseline']['text'].get(virtue, '(no valid answer)')}",
                f"- Few-shot: {first['few_shot']['text'].get(virtue, '(no valid answer)')}"]
    return "\n".join(out)


if __name__ == "__main__":
    report = run_comparison()
    Path("assets").mkdir(exist_ok=True)
    Path("assets/specialization_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    replace_section("model_card.md", "specialization", render_markdown(report))
    print(render_markdown(report))
