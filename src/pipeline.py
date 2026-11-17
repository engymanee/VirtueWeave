"""
The whole VirtueWeave run in one function, run_pipeline(), and a JSON trace.

The six steps, always in this order:
  1. guard_input   rule checks on the teacher's sentence (blocked -> stop)
  2. interpret     the model (or keyword parser) builds topic/age/minutes/description
  3. retrieve      TF-IDF over notes and guides gives each virtue a retrieval score
  4. rank          practices are scored, with a reason for every point
  5. explain       the model words the reasons for the top picks
  6. guard_output  checks the wording; if it fails, the scorer reasons are shown

Every run writes one JSON file to the trace directory, including blocked runs
and runs that crash, so a run can always be replayed from its trace.
"""

import csv
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.explainer import explain, guard_output, group_by_virtue
from src.interpret import Interpretation, interpret, load_topics
from src.llm import LLMUnavailable, call_json
from src.retrieval import VirtueRetriever

TOP_K = 3

# Ranking points. A practice scores topic + fit + retrieval, each with its reason.
TOPIC_POINTS = 2.0       # the practice's topic is the one the teacher named
FIT_POINTS = 1.0         # times the catalog's fit value (0 to 1)
RETRIEVAL_POINTS = 3.0   # times the virtue's retrieval score (0 to 1)

MAX_INPUT_CHARS = 500

INJECTION_PATTERNS = [
    r"ignore (?:all |any |the )?(?:previous|prior|above|earlier) (?:instructions|rules|prompts?)",
    r"disregard (?:all |any |the )?(?:previous|prior|above|earlier)",
    r"system prompt",
    r"you are now\b",
    r"reveal .{0,20}(?:instructions|prompt)",
]


@dataclass
class Pick:
    practice: Dict[str, Any]
    score: float
    reasons: List[str]
    evidence: List[Dict[str, str]]          # {"source", "label", "text"}


@dataclass
class RunResult:
    ok: bool
    sentence: str
    message: str = ""                        # why the run stopped, if it did
    interpretation: Optional[Interpretation] = None
    picks: List[Pick] = field(default_factory=list)
    explained: bool = False                  # True if the model's wording was shown
    virtue_explanations: Dict[str, str] = field(default_factory=dict)
    trace_path: Optional[str] = None


# ------------------------------------------------------------------ step 1

def guard_input(sentence: str) -> Optional[str]:
    """Returns a reason to block the sentence, or None if it is fine."""
    text = sentence.strip()
    if not text:
        return "The sentence is empty. Describe your class in a sentence."
    if len(text) > MAX_INPUT_CHARS:
        return f"The sentence is {len(text)} characters; the limit is {MAX_INPUT_CHARS}."
    if not any(ch.isalpha() for ch in text):
        return "The sentence has no words in it."
    lowered = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return "The sentence looks like an instruction to the assistant, not a classroom."
    return None


# ------------------------------------------------------------------ step 3

def retrieve(retriever: VirtueRetriever, interpretation: Interpretation) -> Dict:
    """Retrieval score and evidence for every virtue."""
    return retriever.match_virtues(interpretation.description)


# ------------------------------------------------------------------ step 4

def load_practices(practices_path: str) -> List[Dict[str, Any]]:
    practices = []
    with open(practices_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            practices.append({
                "id": int(row["id"]), "title": row["title"],
                "virtue": row["virtue"].strip().lower(),
                "topic": row["topic"].strip().lower(),
                "fit": float(row["fit"]), "minutes": int(row["minutes"]),
                "age_min": int(row["age_min"]), "age_max": int(row["age_max"]),
            })
    return practices


def load_notes(notes_path: str) -> Dict[int, Dict[str, str]]:
    with open(notes_path, newline="", encoding="utf-8") as f:
        return {int(r["id"]): {"why": r["why"], "how": r["how"]} for r in csv.DictReader(f)}


def rank(
    practices: List[Dict[str, Any]],
    interpretation: Interpretation,
    matches: Dict,
    notes: Dict[int, Dict[str, str]],
    k: int = TOP_K,
) -> List[Pick]:
    """
    Scores each practice, drops those that don't fit the age or the time, and
    returns the top k. Each reason names the points it earned.
    """
    picks = []
    for p in practices:
        age, minutes = interpretation.age, interpretation.minutes
        if age is not None and not p["age_min"] <= age <= p["age_max"]:
            continue
        if minutes is not None and p["minutes"] > minutes:
            continue

        reasons = []
        score = 0.0
        if interpretation.topic and p["topic"] == interpretation.topic:
            score += TOPIC_POINTS
            reasons.append(f"topic match: {p['topic']} (+{TOPIC_POINTS:.1f})")
        earned = FIT_POINTS * p["fit"]
        score += earned
        reasons.append(f"catalog fit {p['fit']:.1f} for {p['virtue']} in {p['topic']} (+{earned:.2f})")

        match = matches.get(p["virtue"])
        if match is not None:
            earned = RETRIEVAL_POINTS * match.score
            score += earned
            if match.method == "tfidf":
                detail = f"retrieval {match.score:.2f} for {p['virtue']} (text match)"
            else:
                tags = ", ".join(match.cue_hits) or "none"
                detail = f"cue tags for {p['virtue']}: {tags} ({match.score:.2f})"
            reasons.append(f"{detail} (+{earned:.2f})")
        if age is not None:
            reasons.append(f"ages {p['age_min']}-{p['age_max']} include {age}")
        if minutes is not None:
            reasons.append(f"{p['minutes']} min fits in {minutes} min")

        evidence = []
        note = notes.get(p["id"])
        if note:
            for key in ("why", "how"):
                evidence.append({"source": "note", "label": f"{p['title']} ({key})", "text": note[key]})
        if match is not None:
            seen = {e["text"] for e in evidence}
            for hit in match.evidence:
                if hit.passage.text not in seen:
                    evidence.append({
                        "source": hit.passage.source,
                        "label": hit.passage.label,
                        "text": hit.passage.text,
                    })
        picks.append(Pick(p, score, reasons, evidence))

    picks.sort(key=lambda pick: (-pick.score, pick.practice["id"]))
    return picks[:k]


# ------------------------------------------------------------------ the run

def run_pipeline(
    sentence: str,
    practices_path: str = "data/practices.csv",
    notes_path: str = "data/notes.csv",
    guides_path: Optional[str] = None,
    trace_dir: str = "traces",
    model_call: Callable[..., Dict] = call_json,
    with_examples: bool = True,
) -> RunResult:
    """Runs the six steps in order and writes a JSON trace of the run."""
    run_id = uuid.uuid4().hex[:8]
    started = datetime.now(timezone.utc)
    steps: List[Dict[str, Any]] = []
    result = RunResult(ok=False, sentence=sentence)

    def record(name: str, status: str, t0: float, **detail: Any) -> None:
        steps.append({"step": name, "status": status,
                      "ms": round((time.perf_counter() - t0) * 1000, 1), **detail})

    try:
        # 1. input guardrail
        t0 = time.perf_counter()
        blocked = guard_input(sentence)
        if blocked:
            record("guard_input", "blocked", t0, reason=blocked)
            result.message = blocked
            return result
        record("guard_input", "ok", t0)

        # 2. interpret
        t0 = time.perf_counter()
        topics = load_topics(practices_path)
        interpretation = interpret(sentence, topics, model_call)
        result.interpretation = interpretation
        record("interpret", "ok", t0, **asdict(interpretation))

        # 3. retrieve
        t0 = time.perf_counter()
        retriever = VirtueRetriever.from_files(notes_path, practices_path, guides_path)
        matches = retrieve(retriever, interpretation)
        record("retrieve", "ok", t0, query=interpretation.description, virtues={
            v: {"score": round(m.score, 3), "method": m.method,
                "evidence": [h.passage.label for h in m.evidence]}
            for v, m in matches.items()
        })

        # 4. rank
        t0 = time.perf_counter()
        picks = rank(load_practices(practices_path), interpretation, matches,
                     load_notes(notes_path))
        result.picks = picks
        record("rank", "ok" if picks else "empty", t0, picks=[
            {"id": p.practice["id"], "title": p.practice["title"],
             "score": round(p.score, 3), "reasons": p.reasons} for p in picks
        ])
        if not picks:
            result.message = "No practice fits that age and time. Try a longer lesson or a different age."
            return result

        # 5. explain (two or three sentences per virtue)
        t0 = time.perf_counter()
        explanations: Dict[str, str] = {}
        explain_error = None
        try:
            explanations = explain(sentence, picks, matches, model_call, with_examples)
            record("explain", "ok", t0, with_examples=with_examples, explanations=explanations)
        except (LLMUnavailable, ValueError) as error:
            explain_error = str(error)
            record("explain", "failed", t0, reason=explain_error)

        # 6. output guardrail
        t0 = time.perf_counter()
        if explain_error:
            record("guard_output", "skipped", t0, reason="no explanation to check")
        else:
            problem = guard_output(explanations, picks)
            if problem:
                record("guard_output", "failed", t0, reason=problem)
            else:
                result.virtue_explanations = explanations
                result.explained = True
                record("guard_output", "pass", t0)

        result.ok = True
        return result
    except Exception as error:
        steps.append({"step": "crash", "status": "error",
                      "reason": f"{type(error).__name__}: {error}"})
        result.message = f"The run failed: {error}"
        raise
    finally:
        result.trace_path = _write_trace(trace_dir, run_id, started, sentence, steps, result)


def _write_trace(
    trace_dir: str, run_id: str, started: datetime, sentence: str,
    steps: List[Dict[str, Any]], result: RunResult,
) -> str:
    folder = Path(trace_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"run-{started.strftime('%Y%m%d-%H%M%S')}-{run_id}.json"
    trace = {
        "run_id": run_id,
        "started": started.isoformat(),
        "input": sentence,
        "ok": result.ok,
        "message": result.message,
        "shown": "model explanation" if result.explained else
                 ("scorer reasons only" if result.picks else "nothing"),
        "steps": steps,
    }
    path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
    return str(path)


# ------------------------------------------------------------------ display

def format_result(result: RunResult) -> str:
    """Plain-text answer for the terminal."""
    if not result.ok or not result.picks:
        return result.message or "No result."
    i = result.interpretation
    lines = [
        f"Topic: {i.topic or 'not found'} | age: {i.age or '?'} | minutes: {i.minutes or '?'} "
        f"| read by: {i.source}",
        f"Situation: {i.description}",
        "",
    ]
    if not result.explained:
        lines.append("(Showing scoring reasons only.)\n")
    rank_no = 0
    for virtue, group in group_by_virtue(result.picks).items():
        if result.explained:
            lines.append(f"{virtue.title()}: {result.virtue_explanations[virtue]}")
        for pick in group:
            rank_no += 1
            p = pick.practice
            lines.append(f"  #{rank_no}  {p['title']}  ({p['virtue']}, {p['topic']}, {p['minutes']} min, "
                         f"ages {p['age_min']}-{p['age_max']})  score {pick.score:.2f}")
            if not result.explained:
                lines.extend(f"      - {reason}" for reason in pick.reasons)
            lines.append("      Sources: " + "; ".join(e["label"] for e in pick.evidence))
        lines.append("")
    lines.append(f"Trace: {result.trace_path}")
    return "\n".join(lines)
