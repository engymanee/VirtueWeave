"""
Turns the teacher's one sentence into a small request: topic, age, minutes and
a description (the retrieval query).

One model call fills in all four. The topic must be one of the catalog's topics
(the distinct topics in practices.csv). If the model call fails or its answer
does not pass the checks, a keyword parser fills in the same fields instead.
The result says which path was used and, for the fallback, why.
"""

import csv
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from src.llm import LLMUnavailable, call_json


@dataclass(frozen=True)
class Interpretation:
    topic: Optional[str]          # a catalog topic, or None if none was found
    age: Optional[int]
    minutes: Optional[int]
    description: str              # the situation, used as the retrieval query
    source: str                   # "model" or "keyword"
    fallback_reason: Optional[str] = None


def load_topics(practices_path: str) -> List[str]:
    """The catalog's topics, in file order."""
    with open(practices_path, newline="", encoding="utf-8") as f:
        topics = []
        for row in csv.DictReader(f):
            topic = row["topic"].strip().lower()
            if topic not in topics:
                topics.append(topic)
    return topics


# ---------------------------------------------------------------- model path

SYSTEM_PROMPT = (
    "You read one sentence from a teacher planning a coding lesson for young "
    "children and extract four fields.\n"
    "topic: the coding topic being taught, exactly as written in this list: "
    "{topics}. Use null if none fits.\n"
    "age: the children's age in years as an integer, or null if not stated.\n"
    "minutes: the lesson length in minutes as an integer, or null if not stated.\n"
    "description: the classroom situation in one short plain sentence, without "
    "the topic, age or minutes, written so it can be matched against practice "
    "notes (for example, what the children are doing or struggling with). "
    "Use the teacher's own words where you can."
)


def _schema() -> Dict:
    return {
        "type": "object",
        "properties": {
            "topic": {"type": ["string", "null"]},
            "age": {"type": ["integer", "null"]},
            "minutes": {"type": ["integer", "null"]},
            "description": {"type": "string"},
        },
        "required": ["topic", "age", "minutes", "description"],
        "additionalProperties": False,
    }


def _check_model_answer(data: Dict, topics: List[str]) -> Interpretation:
    """Validates the model's JSON. Raises ValueError saying what is wrong."""
    topic = data.get("topic")
    topic = topic.strip().lower() if isinstance(topic, str) else topic
    if topic not in topics:
        raise ValueError(f"topic {data.get('topic')!r} is not in the catalog")

    numbers = {}
    for key in ("age", "minutes"):
        value = data.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"{key} {value!r} is not a positive whole number")
        numbers[key] = value

    description = data.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description is empty")

    return Interpretation(
        topic=topic, age=numbers["age"], minutes=numbers["minutes"],
        description=description.strip(), source="model",
    )


# ------------------------------------------------------------- keyword path

# Words that point to each catalog topic. A topic with no entry here is still
# found by its own name.
TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "algorithms": [r"algorithm", r"instructions?", r"recipe"],
    "sequencing": [r"sequenc", r"in order", r"step by step"],
    "loops": [r"loops?", r"repeat", r"repeating"],
    "events": [r"events?", r"when .{0,20}(?:tapped|clicked|pressed)"],
    "conditionals": [r"conditional", r"if[- ]then", r"if/then"],
    "variables": [r"variables?", r"counter", r"score"],
    "debugging": [r"debug", r"bugs?", r"fix(?:ing)? (?:it|mistakes|errors)"],
    "testing": [r"test"],
    "decomposition": [r"decompos", r"break(?:ing)? .{0,15} down", r"smaller parts"],
    "patterns": [r"patterns?", r"sorting"],
    "pair programming": [r"pair programming", r"pair[- ]program", r"partner"],
    "sharing a project": [r"sharing (?:a|their|the) project", r"show(?:ing)? (?:off )?(?:a|their|the) project",
                          r"present(?:ing)? (?:a|their|the) project"],
}


def _find_topic(text: str, topics: List[str]) -> Optional[str]:
    """The topic with the most keyword hits; ties go to the earlier catalog topic."""
    lowered = text.lower()
    best, best_hits = None, 0
    for topic in topics:
        patterns = TOPIC_KEYWORDS.get(topic, []) + [re.escape(topic)]
        hits = sum(bool(re.search(rf"\b{p}", lowered)) for p in dict.fromkeys(patterns))
        if hits > best_hits:
            best, best_hits = topic, hits
    return best


def _find_age(text: str) -> Optional[int]:
    match = (
        re.search(r"\b(\d{1,2})[- ]?(?:years?|yrs?)[- ]?(?:olds?)?\b", text, re.I)
        or re.search(r"\bage[sd]?\s*(\d{1,2})\b", text, re.I)
        or re.search(r"\bgrade\s*(\d{1,2})\b", text, re.I)
    )
    if not match:
        return None
    age = int(match.group(1))
    if "grade" in match.group(0).lower():
        age += 5  # US grade n is about n + 5 years old
    return age if age > 0 else None


def _find_minutes(text: str) -> Optional[int]:
    match = re.search(r"\b(\d{1,3})[- ]?(?:minutes?|mins?)\b", text, re.I)
    return int(match.group(1)) if match and int(match.group(1)) > 0 else None


def keyword_interpret(
    text: str, topics: List[str], reason: Optional[str] = None
) -> Interpretation:
    """
    The fallback parser. The description is the teacher's sentence with the
    age and minutes phrases removed, so those words don't skew retrieval.
    """
    description = re.sub(
        r"\b\d{1,3}[- ]?(?:minutes?|mins?)\b|\b\d{1,2}[- ]?(?:years?|yrs?)[- ]?(?:olds?)?\b",
        " ", text, flags=re.I,
    )
    topic = _find_topic(text, topics)
    # Keep only the clauses that describe the situation: drop the one that names
    # the topic and any that only give a time ("tomorrow"). Otherwise the topic
    # words would pull retrieval toward every note about that topic.
    topic_patterns = TOPIC_KEYWORDS.get(topic, []) + [re.escape(topic)] if topic else []
    clauses = []
    for clause in re.split(r"[,;.]", description):
        clause = re.sub(r"\s+", " ", clause).strip()
        if not clause or re.fullmatch(r"(?:tomorrow|today|tonight|next week|this week)", clause, re.I):
            continue
        if any(re.search(rf"\b{p}", clause, re.I) for p in topic_patterns):
            continue
        clauses.append(clause)
    description = ", ".join(clauses)
    return Interpretation(
        topic=topic,
        age=_find_age(text),
        minutes=_find_minutes(text),
        description=description or re.sub(r"\s+", " ", text).strip(),
        source="keyword",
        fallback_reason=reason,
    )


# ------------------------------------------------------------------ main API

def interpret(
    text: str,
    topics: List[str],
    model_call: Callable[..., Dict] = call_json,
) -> Interpretation:
    """
    Interprets the teacher's sentence. Tries the model once, checks the answer,
    and falls back to the keyword parser if the call fails or the check fails.
    """
    text = text.strip()
    if not text:
        raise ValueError("The teacher's sentence is empty.")
    try:
        data = model_call(SYSTEM_PROMPT.format(topics=", ".join(topics)), text, _schema())
        return _check_model_answer(data, topics)
    except LLMUnavailable as error:
        return keyword_interpret(text, topics, reason=f"model unavailable: {error}")
    except ValueError as error:
        return keyword_interpret(text, topics, reason=f"model answer rejected: {error}")


if __name__ == "__main__":
    import sys

    sentence = " ".join(sys.argv[1:]) or (
        "Teaching loops to my 6-year-olds tomorrow, 15 minutes, they keep giving up."
    )
    print(interpret(sentence, load_topics("data/practices.csv")))
