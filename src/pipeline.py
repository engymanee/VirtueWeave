"""
The whole VirtueWeave run in one function, run_pipeline(), and a JSON trace.

The teacher gives a topic, a grade and the minutes they have, and may add a
short note about the class. The answer is the 3 virtues that fit best, each
with how it fits the topic and an activity for that grade and time.

The five steps, always in this order:
  1. guard_input   check the fields and the note (blocked -> stop)
  2. retrieve      TF-IDF over the catalog's text scores each virtue against the note
  3. rank          fit + retrieval score each virtue, with a reason for every point
  4. explain       the model words how each virtue fits (or the catalog's own text is shown)
  5. guard_output  check the wording; if it fails, show the catalog text instead

Every run writes one JSON file to the trace directory, including blocked runs
and runs that crash, so a run can always be replayed from its trace.
"""

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.catalog import Catalog, grade_to_band, load_catalog
from src.explainer import explain, guard_output
from src.agent import run_agent
from src.llm import LLMUnavailable, call_json, call_with_tools
from src.retrieval import VirtueRetriever

TOP_K = 3

# Ranking points: a virtue scores its fit for the topic plus how well the note matches it.
FIT_POINTS = 1.0         # times the catalog's fit value (0 to 1)
RETRIEVAL_POINTS = 3.0   # times the virtue's retrieval score (0 to 1)

MAX_NOTE_CHARS = 300
MAX_MINUTES = 180

INJECTION_PATTERNS = [
    r"ignore (?:all |any |the )?(?:previous|prior|above|earlier) (?:instructions|rules|prompts?)",
    r"disregard (?:all |any |the )?(?:previous|prior|above|earlier)",
    r"system prompt",
    r"you are now\b",
    r"reveal .{0,20}(?:instructions|prompt)",
]


@dataclass
class Request:
    topic: str
    grade: str
    minutes: int
    note: str = ""
    band: str = ""          # filled in by guard_input


@dataclass
class Pick:
    virtue: str
    score: float
    title: str
    why: str                 # the catalog's own "how it fits" text
    activity: str
    activity_minutes: int
    over_time: bool          # True if the activity is longer than the teacher's time
    reasons: List[str]
    evidence: List[str]      # other catalog passages that matched the teacher's note


@dataclass
class RunResult:
    ok: bool
    request: Optional[Request] = None
    message: str = ""                        # why the run stopped, if it did
    picks: List[Pick] = field(default_factory=list)
    explained: bool = False                  # True if the model's wording was shown
    virtue_explanations: Dict[str, str] = field(default_factory=dict)
    agent: Optional[Dict[str, Any]] = None   # set in agent mode: what the agent did
    trace_path: Optional[str] = None


# ------------------------------------------------------------------ step 1

def guard_input(topic: Any, grade: Any, minutes: Any, note: Any, catalog: Catalog):
    """Returns (request, None) if the input is fine, or (None, reason) to block it."""
    topic_text = str(topic or "").strip().lower()
    if topic_text not in catalog.topics:
        return None, f"Choose one of the topics: {', '.join(catalog.topics)}."
    try:
        band = grade_to_band(grade)
    except ValueError:
        return None, "Choose a grade from K to 12."
    if isinstance(minutes, bool):
        return None, "Enter the minutes you have as a whole number."
    try:
        mins = int(minutes)
    except (TypeError, ValueError):
        return None, "Enter the minutes you have as a whole number."
    if not 1 <= mins <= MAX_MINUTES:
        return None, f"Minutes must be between 1 and {MAX_MINUTES}."

    note_text = str(note or "").strip()
    if len(note_text) > MAX_NOTE_CHARS:
        return None, f"The note is {len(note_text)} characters; the limit is {MAX_NOTE_CHARS}."
    lowered = note_text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return None, "The note looks like an instruction to the assistant, not a description of your class."
    return Request(topic_text, str(grade).strip().upper(), mins, note_text, band), None


# ------------------------------------------------------------------ step 3

def rank(request: Request, catalog: Catalog, matches: Dict, k: int = TOP_K) -> List[Pick]:
    """
    Scores every virtue for the topic, puts those whose activity fits the time
    first, and returns the top k. If fewer than k fit the time, the best of the
    rest fill the list and are marked over_time.
    """
    candidates = []
    for virtue in catalog.virtues:
        pair = catalog.pairs[(request.topic, virtue)]
        act = catalog.activity(request.topic, virtue, request.band)
        score = FIT_POINTS * pair.fit
        reasons = [f"fit {pair.fit:.1f} for {virtue} in {request.topic} (+{score:.2f})"]

        match = matches.get(virtue)
        if match is not None and match.score > 0:
            earned = RETRIEVAL_POINTS * match.score
            score += earned
            if match.method == "tfidf":
                detail = f"your note matches {virtue}: retrieval {match.score:.2f}"
            elif match.method == "tfidf+cue-tags":
                detail = (f"your note matches {virtue}: retrieval {match.score:.2f} "
                          f"(text and cue tags: {', '.join(match.cue_hits)})")
            else:
                detail = f"your note has cue tags for {virtue}: {', '.join(match.cue_hits)}"
            reasons.append(f"{detail} (+{earned:.2f})")

        over = act.minutes > request.minutes
        if over:
            reasons.append(f"activity takes {act.minutes} min, you have {request.minutes} min")
        else:
            reasons.append(f"activity takes {act.minutes} min, within your {request.minutes} min")

        # The pair's own why, its activity and the virtue guide reach the explainer
        # separately, so the evidence is only what else in the catalog matched the note.
        evidence = []
        if match is not None:
            for hit in match.evidence:
                p = hit.passage
                relevant = p.topic in (None, request.topic) and p.band in (None, request.band)
                if (relevant and p.source != "guide" and p.text not in (pair.why, act.text)
                        and p.text not in evidence):
                    evidence.append(p.text)
        candidates.append(Pick(
            virtue, score, pair.title, pair.why, act.text, act.minutes,
            over, reasons, evidence))

    order = list(catalog.virtues)
    candidates.sort(key=lambda c: (c.over_time, -c.score, order.index(c.virtue)))
    return candidates[:k]


# ------------------------------------------------------------------ the run

def run_pipeline(
    topic: str,
    grade: Any,
    minutes: Any,
    note: str = "",
    data_dir: str = "data",
    trace_dir: str = "traces",
    model_call: Callable[..., Dict] = call_json,
    with_examples: bool = True,
    agent: bool = False,
    tool_call: Callable[..., Any] = call_with_tools,
) -> RunResult:
    """
    Runs the five steps in order and writes a JSON trace of the run. With
    agent=True the model chooses the three virtues with tools (see src/agent.py),
    falling back to the scorer's choice if it cannot.
    """
    run_id = uuid.uuid4().hex[:8]
    started = datetime.now(timezone.utc)
    steps: List[Dict[str, Any]] = []
    raw_input = {"topic": topic, "grade": grade, "minutes": minutes, "note": note}
    result = RunResult(ok=False)

    def record(name: str, status: str, t0: float, **detail: Any) -> None:
        steps.append({"step": name, "status": status,
                      "ms": round((time.perf_counter() - t0) * 1000, 1), **detail})

    try:
        catalog = load_catalog(data_dir)

        # 1. input guardrail
        t0 = time.perf_counter()
        request, blocked = guard_input(topic, grade, minutes, note, catalog)
        if blocked:
            record("guard_input", "blocked", t0, reason=blocked)
            result.message = blocked
            return result
        result.request = request
        record("guard_input", "ok", t0, **asdict(request))

        # 2. retrieve (only if there is a note to match)
        t0 = time.perf_counter()
        matches: Dict = {}
        retriever = None
        if request.note:
            retriever = VirtueRetriever.from_catalog(catalog)
            matches = retriever.match_virtues(request.note)
        record("retrieve", "ok" if matches else "skipped", t0, query=request.note, virtues={
            v: {"score": round(m.score, 3), "method": m.method, "cue_tags": m.cue_hits}
            for v, m in matches.items() if m.score > 0
        })

        # 3. rank
        t0 = time.perf_counter()
        pool = rank(request, catalog, matches, k=len(catalog.virtues))
        picks = pool[:TOP_K]
        result.picks = picks
        record("rank", "ok", t0, picks=[
            {"virtue": p.virtue, "score": round(p.score, 3), "title": p.title,
             "minutes": p.activity_minutes, "over_time": p.over_time, "reasons": p.reasons}
            for p in picks
        ])

        # 3b. agent mode: the model chooses the three virtues with tools
        if agent:
            t0 = time.perf_counter()
            scorer_top = [p.virtue for p in picks]
            run, failure = None, None
            try:
                run = run_agent(request, catalog, retriever, tool_call)
                failure = run.error
            except LLMUnavailable as error:
                failure = f"model unavailable: {error}"
            if run is not None and run.virtues:
                by_virtue = {p.virtue: p for p in pool}
                picks = [by_virtue[v] for v in run.virtues]
                for p in picks:
                    p.reasons.append(f"chosen by the agent: {run.reasoning}")
                result.picks = picks
                result.agent = {"used": True, "reasoning": run.reasoning, "scorer_top3": scorer_top,
                                "agent_top3": run.virtues,
                                "overlap": len(set(scorer_top) & set(run.virtues))}
                record("agent", "ok", t0, steps=run.steps, **result.agent)
            else:
                result.agent = {"used": False, "fallback_reason": failure}
                record("agent", "failed", t0, reason=failure, steps=run.steps if run else [])

        # 4. explain (two or three sentences per virtue)
        t0 = time.perf_counter()
        explanations: Dict[str, str] = {}
        explain_error = None
        guides = catalog.guides()
        try:
            explanations = explain(request, picks, guides, model_call, with_examples)
            record("explain", "ok", t0, with_examples=with_examples, explanations=explanations)
        except (LLMUnavailable, ValueError) as error:
            explain_error = str(error)
            record("explain", "failed", t0, reason=explain_error)

        # 5. output guardrail
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
        result.trace_path = _write_trace(trace_dir, run_id, started, raw_input, steps, result)


def _write_trace(
    trace_dir: str, run_id: str, started: datetime, raw_input: Dict[str, Any],
    steps: List[Dict[str, Any]], result: RunResult,
) -> str:
    folder = Path(trace_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"run-{started.strftime('%Y%m%d-%H%M%S')}-{run_id}.json"
    trace = {
        "run_id": run_id,
        "started": started.isoformat(),
        "input": raw_input,
        "ok": result.ok,
        "message": result.message,
        "shown": "model explanation" if result.explained else
                 ("catalog text only" if result.picks else "nothing"),
        "steps": steps,
    }
    path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
    return str(path)


# ------------------------------------------------------------------ display

def fit_text(pick: Pick, result: RunResult) -> str:
    """The "how it fits" paragraph: the model's wording if it passed, else the catalog's text."""
    return result.virtue_explanations.get(pick.virtue) or pick.why


def format_result(result: RunResult) -> str:
    """Plain-text answer for the terminal."""
    if not result.ok or not result.picks:
        return result.message or "No result."
    r = result.request
    lines = [
        f"Topic: {r.topic} | grade {r.grade} ({r.band}) | {r.minutes} min"
        + (f" | note: {r.note}" if r.note else ""),
        "",
    ]
    if result.agent:
        if result.agent["used"]:
            lines.append(f"(Chosen by the agent, which agrees with the scorer on "
                         f"{result.agent['overlap']} of 3 virtues. Its reasoning: "
                         f"{result.agent['reasoning']})\n")
        else:
            lines.append(f"(The agent was unavailable: {result.agent['fallback_reason']}. "
                         f"Showing the scorer's choice.)\n")
    if not result.explained:
        lines.append("(Showing the catalog's own text; the written note was unavailable.)\n")
    for n, pick in enumerate(result.picks, start=1):
        flag = f"  [runs {pick.activity_minutes} min, over your {r.minutes}]" if pick.over_time else ""
        lines.append(f"#{n}  {pick.virtue.title()}  (score {pick.score:.2f})")
        lines.append(f"    How it fits: {fit_text(pick, result)}")
        lines.append(f"    Try \"{pick.title}\" ({pick.activity_minutes} min){flag}: {pick.activity}")
        lines.append("")
    lines.append(f"Trace: {result.trace_path}")
    return "\n".join(lines)
