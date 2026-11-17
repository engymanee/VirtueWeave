"""
Compares the explainer with and without few-shot examples on 5 requests.

    python -m src.compare_examples

Each request is interpreted, retrieved and ranked once, so both variants explain
exactly the same picks. Only the examples differ. That is 5 interpret calls and
10 explain calls, and it needs working API credentials: if the explainer cannot
be reached the script stops instead of quietly using a fallback, since a
fallback would make the comparison meaningless.

Results are printed side by side and saved to traces/compare-<time>.json. The
numbers (sentences, words per sentence, guardrail result) are only a rough aid;
whether the tone is warm and plain is for you to judge from the text.
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List

from src.explainer import count_sentences, explain, group_by_virtue, guard_output
from src.interpret import interpret, load_topics
from src.llm import LLMUnavailable, call_json
from src.pipeline import load_notes, load_practices, rank, retrieve
from src.retrieval import VirtueRetriever

REQUESTS = [
    "Teaching loops to my 6-year-olds tomorrow, 15 minutes, they keep giving up.",
    "My 7-year-olds argue over who gets the tablet during pair programming, 10 minutes.",
    "We are debugging with 5 and 6 year olds, 10 min, and some blame their partner when it breaks.",
    "Teaching conditionals to 6-year-olds, 10 minutes, a few kids tease classmates who get it wrong.",
    "Sharing projects with parents on Friday, 20 minutes, ages 6, the kids never say thank you.",
]


def metrics(explanations: Dict[str, str], picks: List, ) -> Dict:
    texts = list(explanations.values())
    words = [len(t.split()) for t in texts]
    sentences = [count_sentences(t) for t in texts]
    titles = {v: [p.practice["title"] for p in g] for v, g in group_by_virtue(picks).items()}
    return {
        "guardrail": guard_output(explanations, picks) or "pass",
        "sentences": sentences,
        "words": words,
        "words_per_sentence": round(sum(words) / max(1, sum(sentences)), 1),
        "names_a_practice": sum(
            any(f'"{t}"' in explanations.get(v, "") for t in ts) for v, ts in titles.items()
        ),
        "virtues": len(titles),
    }


def run_comparison(
    model_call: Callable[..., Dict] = call_json,
    requests: List[str] = REQUESTS,
    practices_path: str = "data/practices.csv",
    notes_path: str = "data/notes.csv",
    guides_path: str = None,
    out_dir: str = "traces",
) -> Dict:
    retriever = VirtueRetriever.from_files(notes_path, practices_path, guides_path)
    practices, notes = load_practices(practices_path), load_notes(notes_path)
    topics = load_topics(practices_path)

    rows = []
    for sentence in requests:
        interpretation = interpret(sentence, topics, model_call)
        matches = retrieve(retriever, interpretation)
        picks = rank(practices, interpretation, matches, notes)
        row = {"request": sentence, "interpreted_by": interpretation.source,
               "picks": [p.practice["title"] for p in picks]}
        for label, with_examples in (("without_examples", False), ("with_examples", True)):
            try:
                text = explain(sentence, picks, matches, model_call, with_examples)
            except LLMUnavailable as error:
                raise SystemExit(f"Stopped: the explainer is unavailable ({error}).")
            row[label] = {"text": text, **metrics(text, picks)}
        rows.append(row)

    summary = {}
    for label in ("without_examples", "with_examples"):
        group = [r[label] for r in rows]
        summary[label] = {
            "guardrail_passes": sum(g["guardrail"] == "pass" for g in group),
            "avg_words_per_sentence": round(sum(g["words_per_sentence"] for g in group) / len(group), 1),
            "avg_words_per_virtue": round(
                sum(sum(g["words"]) / len(g["words"]) for g in group) / len(group), 1),
            "practice_named": f"{sum(g['names_a_practice'] for g in group)}/{sum(g['virtues'] for g in group)}",
        }
    report = {"when": datetime.now(timezone.utc).isoformat(), "rows": rows, "summary": summary}
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"compare-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["path"] = str(path)
    return report


def print_report(report: Dict) -> None:
    for i, row in enumerate(report["rows"], start=1):
        print(f"\n=== {i}. {row['request']}")
        print(f"Picks: {', '.join(row['picks'])}  (read by {row['interpreted_by']})")
        for label in ("without_examples", "with_examples"):
            block = row[label]
            print(f"\n  -- {label.replace('_', ' ')}  [guardrail: {block['guardrail']}, "
                  f"{block['words_per_sentence']} words/sentence]")
            for virtue, text in block["text"].items():
                print(f"  {virtue.title()}: {text}")
    print("\n=== Summary")
    for label, s in report["summary"].items():
        print(f"  {label.replace('_', ' '):17s} {s}")
    print(f"\nSaved to {report['path']}")


if __name__ == "__main__":
    print_report(run_comparison())
    sys.exit(0)
