"""
TF-IDF retrieval over the practice notes (why and how) and the virtue guides.

The teacher's description is the query. Each passage is scored by cosine
similarity, and each virtue gets a retrieval score: the best score among its
passages (0 to 1). The matching passages travel with the score as evidence for
the explainer. Cue tags are a plain keyword backup, used only when the text
search finds nothing.

Files:
  data/practices.csv  id, title, virtue, ..., cue_tags (tags split on "|")
  data/notes.csv      id, why, how            (one row per practice id)
  guides CSV          virtue, text[, title]   (optional, one row per passage)
"""

import csv
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


@dataclass(frozen=True)
class Passage:
    """One indexed piece of text and where it came from."""
    source: str                    # "note" or "guide"
    virtue: str
    text: str
    label: str                     # e.g. "Steps to Help (why)" or a guide title
    practice_id: Optional[int] = None


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
    method: str                    # "tfidf" or "cue-tags"
    evidence: List[Hit]            # best passages first (empty for cue-tags)
    cue_hits: List[str]            # cue tags found in the description


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_note_passages(notes_path: str, practices_path: str) -> List[Passage]:
    """Builds one passage per why and one per how, tagged with the practice's virtue."""
    practices = {row["id"]: row for row in _read_csv(practices_path)}
    passages = []
    for row in _read_csv(notes_path):
        practice = practices.get(row["id"])
        if practice is None:
            raise ValueError(f"notes.csv id {row['id']} is not in practices.csv")
        for field in ("why", "how"):
            text = (row.get(field) or "").strip()
            if text:
                passages.append(Passage(
                    source="note",
                    virtue=practice["virtue"].strip().lower(),
                    text=text,
                    label=f"{practice['title']} ({field})",
                    practice_id=int(row["id"]),
                ))
    return passages


def load_guide_passages(guides_path: str) -> List[Passage]:
    """Builds one passage per guide row (columns: virtue, text, optional title)."""
    passages = []
    for number, row in enumerate(_read_csv(guides_path), start=1):
        text = (row.get("text") or "").strip()
        virtue = (row.get("virtue") or "").strip().lower()
        if text and virtue:
            title = (row.get("title") or "").strip() or f"{virtue} guide {number}"
            passages.append(Passage("guide", virtue, text, title))
    return passages


def load_cue_tags(practices_path: str) -> Dict[str, List[str]]:
    """Returns each virtue's cue tags (the union over its practices)."""
    tags: Dict[str, List[str]] = {}
    for row in _read_csv(practices_path):
        virtue = row["virtue"].strip().lower()
        bucket = tags.setdefault(virtue, [])
        for tag in (row.get("cue_tags") or "").split("|"):
            tag = tag.strip().lower()
            if tag and tag not in bucket:
                bucket.append(tag)
    return tags


def _words(text: str) -> str:
    """Lowercases and turns punctuation into spaces so "laughing-at" == "laughing at"."""
    return " " + " ".join(re.findall(r"[a-z0-9']+", text.lower())) + " "


class VirtueRetriever:
    """TF-IDF index over passages, queried with the teacher's description."""

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
            stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        )
        self._matrix = self._vectorizer.fit_transform(p.text for p in passages)

    @classmethod
    def from_files(
        cls,
        notes_path: str,
        practices_path: str,
        guides_path: Optional[str] = None,
    ) -> "VirtueRetriever":
        passages = load_note_passages(notes_path, practices_path)
        if guides_path:
            passages += load_guide_passages(guides_path)
        return cls(passages, load_cue_tags(practices_path))

    def search(self, description: str, top_n: int = 5) -> List[Hit]:
        """Returns up to top_n passages with a similarity above 0, best first."""
        if not description.strip():
            return []
        query = self._vectorizer.transform([description])
        scores = linear_kernel(query, self._matrix).ravel()
        order = scores.argsort()[::-1]
        return [
            Hit(self.passages[i], float(scores[i]))
            for i in order[:top_n] if scores[i] > 0
        ]

    def cue_matches(self, description: str) -> Dict[str, List[str]]:
        """Cue tags from each virtue that appear in the description."""
        text = _words(description)
        return {
            virtue: [tag for tag in self.cue_tags.get(virtue, []) if _words(tag) in text]
            for virtue in self.virtues
        }

    def match_virtues(
        self, description: str, evidence_per_virtue: int = 3
    ) -> Dict[str, VirtueMatch]:
        """
        Gives every virtue a retrieval score and its evidence passages.

        The score is the best passage similarity for that virtue. If no passage
        matches at all (for example the description uses none of the indexed
        words), the score falls back to the share of the virtue's cue tags
        found in the description.
        """
        hits = self.search(description, top_n=len(self.passages))
        cues = self.cue_matches(description)
        use_text = bool(hits)

        results = {}
        for virtue in self.virtues:
            mine = [h for h in hits if h.passage.virtue == virtue]
            tags = self.cue_tags.get(virtue, [])
            if use_text:
                score = mine[0].score if mine else 0.0
                method = "tfidf"
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


def format_evidence(match: VirtueMatch) -> str:
    """Formats a virtue's evidence as lines for the explainer's prompt."""
    return "\n".join(
        f'- [{h.passage.source}: {h.passage.label}] "{h.passage.text}" '
        f"(similarity {h.score:.2f})"
        for h in match.evidence
    )


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "two children keep arguing about whose turn it is"
    retriever = VirtueRetriever.from_files("data/notes.csv", "data/practices.csv")
    ranked = sorted(retriever.match_virtues(query).values(), key=lambda m: -m.score)
    print(f"Query: {query}\n")
    for m in ranked:
        print(f"{m.virtue:12s} {m.score:.3f}  ({m.method})")
        if m.evidence:
            print(format_evidence(m))
