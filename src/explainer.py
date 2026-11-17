"""
Step 4 and 5 of the pipeline: the model writes two or three sentences per
virtue saying how it fits the teacher's topic, and a guardrail checks them.

The model is given the catalog's own "why it fits" text, the activity for the
grade band, the virtue guide and the scorer's reasons, so it rewords facts it
was handed and does not invent them. Few-shot examples in the system prompt set
a warm, plain tone. They can be switched off (with_examples=False) to compare.

The call goes through src.llm.call_json like every other model call.
"""

import json
import re
from typing import Any, Callable, Dict, List

MIN_SENTENCES, MAX_SENTENCES = 2, 3
MAX_CHARS = 600

RULES = (
    "You help a computer science teacher bring a virtue into their lesson. For "
    "each virtue you are given, write two or three sentences to the teacher "
    "about how that virtue fits the topic they are teaching.\n"
    "Rules:\n"
    "- Speak to the teacher in a warm, plain voice, like a helpful colleague.\n"
    "- Use short everyday words. No jargon, no lists, no headings, no emoji.\n"
    "- Use only the facts you are given: how_it_fits, the activity, the virtue "
    "guide, the retrieved passages and the scorer's reasons. Do not invent "
    "facts, activities, links or numbers.\n"
    "- Name the activity in double quotes, exactly as titled, and say how it "
    "suits the grade level and the minutes they have.\n"
    "- If the teacher left a note about their class, connect to it. If not, "
    "do not make one up.\n"
    "- Exactly 2 or 3 sentences per virtue. Avoid abbreviations with full stops.\n"
    'Return JSON: {"explanations": [{"virtue": "...", "text": "..."}]}, one item '
    "per virtue given, using the virtue name exactly as given."
)

# Invented topic, virtues and activities, so the model copies the tone and not the content.
FEW_SHOT_EXAMPLES = [
    (
        {
            "topic": "sequencing", "grade": "2", "grade_band": "K-2", "minutes": 10,
            "note": "they rush and get cross when a step goes wrong",
            "virtues": [{
                "virtue": "patience",
                "guide": "Patience is waiting calmly and doing things one at a time.",
                "how_it_fits": "A robot only does the steps it is given, so rushing a step makes it fail.",
                "activity": {"title": "Slow Robot", "minutes": 10,
                             "text": "One child gives one instruction at a time and the robot carries it out."},
                "reasons": ["topic fit 0.8 for patience in sequencing (+0.80)",
                            "your note matches patience: retrieval 0.31 (+0.93)"],
            }],
        },
        {"explanations": [{"virtue": "patience", "text": (
            "Rushing is very common at this age, and a robot that does exactly "
            "what it is told shows children why slowing down helps. In "
            '"Slow Robot" they give one instruction at a time and watch it '
            "carried out, so waiting becomes part of the fun. It fits your "
            "sequencing lesson and the ten minutes you have."
        )}]},
    ),
    (
        {
            "topic": "variables", "grade": "9", "grade_band": "9-12", "minutes": 30,
            "note": None,
            "virtues": [{
                "virtue": "courage",
                "guide": "Courage is doing something even when it feels risky.",
                "how_it_fits": "Showing the values a program stores means showing real work, including the parts that are not finished.",
                "activity": {"title": "Show Your Variables", "minutes": 30,
                             "text": "Students walk a partner through their program and explain what each variable holds."},
                "reasons": ["topic fit 0.7 for courage in variables (+0.70)"],
            }],
        },
        {"explanations": [{"virtue": "courage", "text": (
            "Explaining what each variable holds means showing real, unfinished work, "
            "and that takes a little courage. \"Show Your Variables\" lets students "
            "practise it with one partner before any bigger audience. It suits your "
            "thirty minutes and an older group that can explain their thinking."
        )}]},
    ),
]


def build_system_prompt(with_examples: bool) -> str:
    if not with_examples:
        return RULES
    blocks = [
        f"<example>\nInput:\n{json.dumps(given, indent=1)}\nOutput:\n{json.dumps(answer)}\n</example>"
        for given, answer in FEW_SHOT_EXAMPLES
    ]
    return RULES + "\n\nExamples of the tone and length wanted:\n\n" + "\n\n".join(blocks)


def _schema() -> Dict:
    return {
        "type": "object",
        "properties": {
            "explanations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"virtue": {"type": "string"}, "text": {"type": "string"}},
                    "required": ["virtue", "text"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["explanations"],
        "additionalProperties": False,
    }


def build_input(request: Any, picks: List[Any], guides: Dict[str, str]) -> Dict:
    """The user message: the request plus one entry per picked virtue."""
    return {
        "topic": request.topic, "grade": request.grade, "grade_band": request.band,
        "minutes": request.minutes, "note": request.note or None,
        "virtues": [{
            "virtue": p.virtue,
            "guide": guides.get(p.virtue, ""),
            "how_it_fits": p.why,
            "activity": {"title": p.title, "minutes": p.activity_minutes, "text": p.activity},
            "reasons": p.reasons,
            "retrieved_passages": [e["text"] for e in p.evidence if e["source"] != "guide"][:4],
        } for p in picks],
    }


def explain(
    request: Any,
    picks: List[Any],
    guides: Dict[str, str],
    model_call: Callable[..., Dict],
    with_examples: bool = True,
) -> Dict[str, str]:
    """One model call. Returns {virtue: text}. Raises ValueError if the shape is wrong."""
    data = model_call(
        build_system_prompt(with_examples),
        json.dumps(build_input(request, picks, guides)),
        _schema(),
    )
    items = data.get("explanations")
    if not isinstance(items, list):
        raise ValueError("no explanations list")
    return {
        item["virtue"].strip().lower(): item["text"]
        for item in items
        if isinstance(item, dict)
        and isinstance(item.get("virtue"), str) and isinstance(item.get("text"), str)
    }


def count_sentences(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])


def guard_output(explanations: Dict[str, str], picks: List[Any]) -> str:
    """Returns a reason the wording failed, or "" if it passes."""
    wanted = {p.virtue for p in picks}
    if set(explanations) != wanted:
        return f"explanations cover {sorted(explanations)}, expected {sorted(wanted)}"
    for virtue, text in explanations.items():
        sentences = count_sentences(text)
        if not MIN_SENTENCES <= sentences <= MAX_SENTENCES:
            return f"{virtue}: {sentences} sentences (want {MIN_SENTENCES}-{MAX_SENTENCES})"
        if len(text) > MAX_CHARS:
            return f"{virtue}: {len(text)} characters (limit {MAX_CHARS})"
        if re.search(r"https?://|www\.|@\w", text):
            return f"{virtue}: contains a link or address"
    return ""
