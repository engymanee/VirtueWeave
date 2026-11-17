"""
Agent mode: instead of ranking by a fixed formula, the model chooses the three
virtues by calling tools, and every step is recorded.

The agent can
  get_topic_fits         see how well each of the 8 virtues fits the topic
  search_class_notes     match the teacher's note against the catalog text
  get_activity           read the activity for a virtue at this grade and check it fits the time
  submit_recommendation  hand in exactly 3 distinct virtues with its reasoning

Every tool call carries a short "reason", so the saved trace shows what the agent
was thinking at each step. The submission is validated, and the pipeline compares
the agent's choice with the scorer's, so a disagreement is visible. If the model is
unavailable or never submits a valid answer, the caller falls back to the scorer.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.catalog import Catalog
from src.llm import call_with_tools

MAX_STEPS = 8
MAX_BAD_SUBMISSIONS = 2

SYSTEM_PROMPT = (
    "You choose the three virtues that best fit a computer science lesson, for a "
    "teacher. Work in steps and use the tools:\n"
    "1. Call get_topic_fits to see how well each virtue fits the topic.\n"
    "2. If the teacher left a note, call search_class_notes with that note to see "
    "which virtues it points to. Skip this if there is no note.\n"
    "3. Call get_activity for the virtues you are considering, to check the activity "
    "for the grade and whether it fits the minutes available. Prefer virtues whose "
    "activity fits the time.\n"
    "4. Call submit_recommendation with exactly 3 distinct virtues, best first.\n"
    "Every tool call needs a short reason. Use only what the tools return, and do not "
    "invent virtues, activities or numbers."
)

_REASON = {"type": "string", "description": "One short sentence: why you are calling this tool now."}


def _schema(**properties: Any) -> Dict[str, Any]:
    props = {**properties}
    return {"type": "object", "properties": props, "required": list(props),
            "additionalProperties": False}


TOOLS: List[Dict[str, Any]] = [
    {"name": "get_topic_fits",
     "description": "How well each of the 8 virtues fits the topic, with a one-sentence why for each.",
     "input_schema": _schema(reason=_REASON)},
    {"name": "search_class_notes",
     "description": "Match the teacher's note against the catalog. Returns the virtues the note points to, with scores.",
     "input_schema": _schema(query={"type": "string", "description": "The teacher's note."}, reason=_REASON)},
    {"name": "get_activity",
     "description": "The activity for one virtue at the teacher's grade, with its minutes and whether it fits the time.",
     "input_schema": _schema(virtue={"type": "string"}, reason=_REASON)},
    {"name": "submit_recommendation",
     "description": "Hand in the final answer: exactly 3 distinct virtues, best first.",
     "input_schema": _schema(
         virtues={"type": "array", "items": {"type": "string"}, "description": "Exactly 3 virtue names."},
         reasoning={"type": "string", "description": "Two or three sentences on why these three."})},
]


@dataclass
class AgentRun:
    virtues: Optional[List[str]] = None       # the 3 submitted, or None if it never submitted
    reasoning: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None               # why there is no valid answer


class _Tools:
    """Runs the agent's tools against the catalog."""

    def __init__(self, request: Any, catalog: Catalog, retriever: Optional[Any]):
        self.request, self.catalog, self.retriever = request, catalog, retriever
        self.accepted: Optional[List[str]] = None
        self.reasoning = ""
        self.bad_submissions = 0

    def run(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        handler = getattr(self, f"_{name}", None) if name in {t["name"] for t in TOOLS} else None
        if handler is None:
            return {"error": f"unknown tool {name!r}"}
        try:
            return handler(args)
        except (KeyError, TypeError, ValueError) as error:
            return {"error": f"bad arguments: {error}"}

    def _get_topic_fits(self, args):
        r = self.request
        return {"topic": r.topic, "fits": {
            v: {"fit": self.catalog.pairs[(r.topic, v)].fit, "why": self.catalog.pairs[(r.topic, v)].why}
            for v in self.catalog.virtues}}

    def _search_class_notes(self, args):
        if self.retriever is None or not str(args.get("query", "")).strip():
            return {"matches": [], "message": "no note to search"}
        matches = self.retriever.match_virtues(str(args["query"]))
        best = sorted((m for m in matches.values() if m.score > 0), key=lambda m: -m.score)[:4]
        return {"matches": [{"virtue": m.virtue, "score": round(m.score, 2), "cue_tags": m.cue_hits}
                            for m in best]}

    def _get_activity(self, args):
        virtue = str(args["virtue"]).strip().lower()
        if virtue not in self.catalog.virtues:
            raise ValueError(f"unknown virtue {virtue!r}")
        r = self.request
        act = self.catalog.activity(r.topic, virtue, r.band)
        return {"virtue": virtue, "title": self.catalog.pairs[(r.topic, virtue)].title,
                "minutes": act.minutes, "fits_time": act.minutes <= r.minutes,
                "minutes_available": r.minutes, "activity": act.text}

    def _submit_recommendation(self, args):
        virtues = [str(v).strip().lower() for v in args["virtues"]]
        reasoning = str(args.get("reasoning", "")).strip()
        problem = None
        if len(virtues) != 3 or len(set(virtues)) != 3:
            problem = "submit exactly 3 distinct virtues"
        elif any(v not in self.catalog.virtues for v in virtues):
            problem = f"every virtue must be one of: {', '.join(self.catalog.virtues)}"
        elif not reasoning:
            problem = "give your reasoning"
        if problem:
            self.bad_submissions += 1
            return {"accepted": False, "error": problem}
        self.accepted, self.reasoning = virtues, reasoning
        return {"accepted": True}


def _short(result: Dict[str, Any], limit: int = 500) -> str:
    text = json.dumps(result)
    return text if len(text) <= limit else text[:limit] + "..."


def run_agent(
    request: Any,
    catalog: Catalog,
    retriever: Optional[Any] = None,
    tool_call: Callable[..., Any] = call_with_tools,
    max_steps: int = MAX_STEPS,
) -> AgentRun:
    """
    Runs the tool loop. Raises LLMUnavailable if the model cannot be reached; any
    other problem (no valid submission, too many steps) is returned in AgentRun.error.
    """
    tools = _Tools(request, catalog, retriever)
    task = {"topic": request.topic, "grade": request.grade, "grade_band": request.band,
            "minutes_available": request.minutes, "teacher_note": request.note or None}
    messages: List[Dict[str, Any]] = [{"role": "user", "content": json.dumps(task)}]
    run = AgentRun()

    for _ in range(max_steps):
        response = tool_call(SYSTEM_PROMPT, messages, TOOLS)
        said = " ".join(b.text.strip() for b in response.content
                        if b.type == "text" and b.text.strip())
        uses = [b for b in response.content if b.type == "tool_use"]
        if not uses:
            run.error = "the agent stopped without submitting a recommendation"
            return run

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for use in uses:
            args = dict(use.input or {})
            output = tools.run(use.name, args)
            run.steps.append({
                "step": len(run.steps) + 1,
                "said": said,
                "tool": use.name,
                "reason": args.get("reason") or args.get("reasoning") or "",
                "input": {k: v for k, v in args.items() if k not in ("reason", "reasoning")},
                "result": _short(output),
            })
            said = ""
            results.append({"type": "tool_result", "tool_use_id": use.id, "content": json.dumps(output)})
        messages.append({"role": "user", "content": results})

        if tools.accepted:
            run.virtues, run.reasoning = tools.accepted, tools.reasoning
            return run
        if tools.bad_submissions > MAX_BAD_SUBMISSIONS:
            run.error = "the agent kept submitting invalid recommendations"
            return run

    run.error = f"the agent did not submit within {max_steps} steps"
    return run
