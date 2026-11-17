"""
TF-IDF retrieval over the catalog's text: virtue guides, "why it fits" notes and
activities.

The teacher's optional note is the query. Each passage is scored by cosine
similarity, and each virtue gets a retrieval score: the best score among its
passages (0 to 1), plus a bonus when one of the virtue's cue tags appears in
the note. The matching passages travel with the score as evidence.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src.catalog import Catalog


@dataclass(frozen=True)
class Passage:
    """One indexed piece of text and where it came from."""
    source: str                    # "guide", "why" or "activity"
    virtue: str
    text: str
    label: str                     # e.g. "Fix the Recipe (why)"
    topic: Optional[str] = None
    band: Optional[str] = None


@dataclass(frozen=True)
class Hit:
    """A passage that matched the query, with its cosine similarity."""
    passage: Passage
    score: float


@dataclass(frozen=True)
class VirtueMatch:
    """Retrieval result for one virtue."""
    virtue: str
    score: float                   # 0 to 1
    method: str                    # "tfidf", "tfidf+cue-tags" or "cue-tags"
    evidence: List[Hit]            # best passages first
    cue_hits: List[str]            # cue tags found in the note


def passages_from_catalog(catalog: Catalog) -> List[Passage]:
    """Every virtue guide, every why and every activity in the catalog."""
    passages = [
        Passage("guide", v.name, v.guide, f"{v.name.title()} guide")
        for v in catalog.virtues.values()
    ]
    passages += [
        Passage("why", p.virtue, p.why, f"{p.title} (why)", topic=p.topic)
        for p in catalog.pairs.values()
    ]
    passages += [
        Passage("activity", a.virtue, a.text,
                f"{catalog.pairs[(a.topic, a.virtue)].title} ({a.band} activity)",
                topic=a.topic, band=a.band)
        for a in catalog.activities.values()
    ]
    return passages


# Added to a virtue's text-match score when any of its cue tags appears in the
# note. Tags are specific words ("blaming", "teasing"), so one is enough.
CUE_BONUS = 0.3

# Text matches weaker than this are treated as chance overlap of a word or two and
# ignored. Cue tags are not affected.
MIN_SIMILARITY = 0.25

_SUFFIXES = ("ing", "ed", "es", "s", "ly")


def stem(word: str) -> str:
    """
    A tiny stemmer so "tease", "teasing" and "teases" meet at "teas", and
    "blame" meets "blaming". It only needs to be consistent, not correct.
    """
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if word.endswith("e") and len(word) > 3:
        word = word[:-1]
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "lsz":
        word = word[:-1]            # running -> runn -> run
    return word


# Words every classroom note and nearly every passage contains. They say nothing
# about which virtue fits, and on a one-word note they would match by accident.
DOMAIN_STOP_WORDS = {
    "class", "classes", "classroom", "student", "students", "child", "children",
    "kid", "kids", "teacher", "teachers", "lesson", "lessons", "today", "tomorrow",
    "week", "day", "days", "year", "years", "old", "thing", "things", "group",
    "groups", "everyone", "program", "programs", "programming", "code",
}
_STOP_WORDS = set(ENGLISH_STOP_WORDS) | DOMAIN_STOP_WORDS


def _tokens(text: str) -> List[str]:
    return [stem(w) for w in re.findall(r"[a-z0-9']+", text.lower())
            if w not in _STOP_WORDS]


def _words(text: str) -> str:
    """Stemmed words joined by spaces, so "laughing-at" matches "laughs at"."""
    stems = [stem(w) for w in re.findall(r"[a-z0-9']+", text.lower())]
    return " " + " ".join(stems) + " "


class VirtueRetriever:
    """TF-IDF index over passages, queried with the teacher's note."""

    def __init__(
        self,
        passages: List[Passage],
        cue_tags: Optional[Dict[str, List[str]]] = None,
    ):
        if not passages:
            raise ValueError("Cannot build a retrieval index with no passages.")
        self.passages = passages
        self.cue_tags = cue_tags or {}
        self.virtues = sorted({p.virtue for p in passages} | set(self.cue_tags))
        self._vectorizer = TfidfVectorizer(
            tokenizer=_tokens, token_pattern=None, lowercase=False,
            ngram_range=(1, 2), sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(p.text for p in passages)

    @classmethod
    def from_catalog(cls, catalog: Catalog) -> "VirtueRetriever":
        return cls(
            passages_from_catalog(catalog),
            {name: info.cue_tags for name, info in catalog.virtues.items()},
        )

    def search(self, note: str, top_n: int = 5) -> List[Hit]:
        """Returns up to top_n passages with a similarity above 0, best first."""
        if not note.strip():
            return []
        query = self._vectorizer.transform([note])
        scores = linear_kernel(query, self._matrix).ravel()
        order = scores.argsort(kind="stable")[::-1]
        return [
            Hit(self.passages[i], float(scores[i]))
            for i in order[:top_n] if scores[i] > 0
        ]

    def cue_matches(self, note: str) -> Dict[str, List[str]]:
        """Cue tags from each virtue that appear in the note."""
        text = _words(note)
        return {
            virtue: [tag for tag in self.cue_tags.get(virtue, []) if _words(tag) in text]
            for virtue in self.virtues
        }

    def match_virtues(
        self, note: str, evidence_per_virtue: int = 3
    ) -> Dict[str, VirtueMatch]:
        """
        Gives every virtue a retrieval score and its evidence passages.

        Only text matches of at least MIN_SIMILARITY count. When one does, a virtue's score is its best
        passage similarity plus CUE_BONUS if any of its cue tags is found in
        the note, capped at 1. If no passage matches at all, the score is the
        share of the virtue's cue tags found in the note.
        """
        hits = [h for h in self.search(note, top_n=len(self.passages))
                if h.score >= MIN_SIMILARITY]
        cues = self.cue_matches(note)
        use_text = bool(hits)

        results = {}
        for virtue in self.virtues:
            mine = [h for h in hits if h.passage.virtue == virtue]
            tags = self.cue_tags.get(virtue, [])
            if use_text:
                text_score = mine[0].score if mine else 0.0
                bonus = CUE_BONUS if cues[virtue] else 0.0
                score = min(1.0, text_score + bonus)
                method = "tfidf+cue-tags" if cues[virtue] else "tfidf"
            else:
                score = len(cues[virtue]) / len(tags) if tags else 0.0
                method = "cue-tags"
            results[virtue] = VirtueMatch(
                virtue=virtue,
                score=score,
                method=method,
                evidence=mine[:evidence_per_virtue],
                cue_hits=cues[virtue],
            )
        return results
