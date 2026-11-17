"""
Step 5 and 6 of the pipeline: the model writes two or three sentences per
virtue for the teacher, and a guardrail checks what it wrote.

Input to the model, per virtue: the retrieved passages (practice notes and
guides), the scorer's reasons and the picked practices. Few-shot examples in
the system prompt set a warm, plain tone. They can be switched off
(with_examples=False) to compare. Both variants carry the same rules, so the
only difference is the examples.

The call goes through src.llm.call_json like every other model call.
"""

import json
import re
from typing import Any, Callable, Dict, List

MIN_SENTENCES, MAX_SENTENCES = 2, 3
MAX_CHARS = 600

RULES = (
    "You help a teacher of young children choose coding-lesson practices that "
    "also build a virtue. For each virtue you are given, write two or three "
    "sentences to the teacher about why it suits their class.\n"
    "Rules:\n"
    "- Speak to the teacher in a warm, plain voice, like a helpful colleague.\n"
    "- Use short everyday words. No jargon, no lists, no headings, no emoji.\n"
    "- Use only the scorer's reasons and the retrieved passages you are given. "
    "Do not invent facts, activities, links or numbers.\n"
    "- Name each practice you mention in double quotes, exactly as titled.\n"
    "- Exactly 2 or 3 sentences per virtue. Avoid abbreviations with full stops.\n"
    'Return JSON: {"explanations": [{"virtue": "...", "text": "..."}]}, one item '
    "per virtue given, using the virtue name exactly as given."
)

# Invented virtues and practices, so the model copies the tone and not the content.
FEW_SHOT_EXAMPLES = [
    (
        {
            "teacher_said": "Sequencing with my 7-year-olds, they rush and get cross when a step goes wrong.",
            "virtues": [{
                "virtue": "patience",
                "retrieval": {"score": 0.31, "method": "tfidf"},
                "practices": [{
                    "title": "Slow Robot", "minutes": 10, "ages": "5-7",
                    "reasons": ["topic match: sequencing (+2.0)", "retrieval 0.31 for patience (text match)"],
                }],
                "retrieved_passages": [
                    "[Slow Robot (why)] A robot only does the steps it is given, so rushing a step makes it fail.",
                    "[Slow Robot (how)] One child gives one instruction at a time and the robot carries it out.",
                ],
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
            "teacher_said": "Variables with 6-year-olds, 15 minutes, a few are shy about showing their work.",
            "virtues": [{
                "virtue": "courage",
                "retrieval": {"score": 0.0, "method": "cue-tags"},
                "practices": [{
                    "title": "Show One Thing", "minutes": 10, "ages": "5-7",
                    "reasons": ["topic match: variables (+2.0)", "cue tags for courage: shy (0.17)"],
                }],
                "retrieved_passages": [
                    "[Show One Thing (how)] Each child shows a partner one thing they changed in their program.",
                ],
            }],
        },
        {"explanations": [{"virtue": "courage", "text": (
            "Your shy children may find it easier to share when they only have "
            'to show one small thing. "Show One Thing" asks each child to tell '
            "a partner what they changed, which is a gentle first step before "
            "a bigger audience. It suits variables and fits inside your fifteen minutes."
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


def group_by_virtue(picks: List[Any]) -> Dict[str, List[Any]]:
    """Picks grouped by virtue, in the order each virtue first appears."""
    groups: Dict[str, List[Any]] = {}
    for pick in picks:
        groups.setdefault(pick.practice["virtue"], []).append(pick)
    return groups


def build_input(sentence: str, picks: List[Any], matches: Dict) -> Dict:
    """The user message: one entry per virtue with practices, reasons and passages."""
    virtues = []
    for virtue, group in group_by_virtue(picks).items():
        passages, seen = [], set()
        for pick in group:
            for e in pick.evidence:
                if e["text"] not in seen:
                    seen.add(e["text"])
                    passages.append(f"[{e['label']}] {e['text']}")
        match = matches.get(virtue)
        virtues.append({
            "virtue": virtue,
            "retrieval": {"score": round(match.score, 2), "method": match.method} if match else None,
            "practices": [{
                "title": p.practice["title"], "minutes": p.practice["minutes"],
                "ages": f"{p.practice['age_min']}-{p.practice['age_max']}",
                "reasons": p.reasons,
            } for p in group],
            "retrieved_passages": passages[:6],
        })
    return {"teacher_said": sentence, "virtues": virtues}


def explain(
    sentence: str,
    picks: List[Any],
    matches: Dict,
    model_call: Callable[..., Dict],
    with_examples: bool = True,
) -> Dict[str, str]:
    """One model call. Returns {virtue: text}. Raises ValueError if the shape is wrong."""
    data = model_call(
        build_system_prompt(with_examples),
        json.dumps(build_input(sentence, picks, matches)),
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
    wanted = set(group_by_virtue(picks))
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
