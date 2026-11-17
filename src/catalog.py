"""
The catalog: what VirtueWeave knows.

  data/virtues.csv      virtue, meaning, guide (what the virtue is), cue_tags
  data/virtue_fit.csv   topic, virtue, fit (0-1), title, why (how the virtue fits the topic)
  data/activities.csv   topic, virtue, band, minutes, activity (one per grade band)

Every topic has an entry for every virtue (12 x 8), and every pair has an
activity for each of the four grade bands, so the app can pick the best 3
virtues for any topic.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

BANDS = ["K-2", "3-5", "6-8", "9-12"]
GRADES = ["K"] + [str(n) for n in range(1, 13)]


def grade_to_band(grade) -> str:
    """Maps a grade (K or 1 to 12) to its band. Raises ValueError for anything else."""
    text = str(grade).strip().upper()
    if text in ("K", "KINDERGARTEN"):
        return "K-2"
    if text.isdigit() and 1 <= int(text) <= 12:
        n = int(text)
        return "K-2" if n <= 2 else "3-5" if n <= 5 else "6-8" if n <= 8 else "9-12"
    raise ValueError(f"grade must be K or 1 to 12, not {grade!r}")


@dataclass(frozen=True)
class VirtueInfo:
    name: str
    meaning: str
    guide: str
    cue_tags: List[str]


@dataclass(frozen=True)
class Pair:
    topic: str
    virtue: str
    fit: float
    title: str
    why: str


@dataclass(frozen=True)
class Activity:
    topic: str
    virtue: str
    band: str
    minutes: int
    text: str


@dataclass
class Catalog:
    topics: List[str]
    virtues: Dict[str, VirtueInfo]
    pairs: Dict[Tuple[str, str], Pair]
    activities: Dict[Tuple[str, str, str], Activity]

    def activity(self, topic: str, virtue: str, band: str) -> Activity:
        return self.activities[(topic, virtue, band)]

    def guides(self) -> Dict[str, str]:
        """Each virtue's guide text, by virtue name."""
        return {name: info.guide for name, info in self.virtues.items()}


def _rows(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_catalog(data_dir: str = "data") -> Catalog:
    """Loads and checks the three files. Raises ValueError if anything is missing."""
    folder = Path(data_dir)
    virtues = {}
    for r in _rows(folder / "virtues.csv"):
        name = r["virtue"].strip().lower()
        tags = [t.strip().lower() for t in r["cue_tags"].split("|") if t.strip()]
        virtues[name] = VirtueInfo(name, r["meaning"].strip(), r["guide"].strip(), tags)

    pairs: Dict[Tuple[str, str], Pair] = {}
    topics: List[str] = []
    for r in _rows(folder / "virtue_fit.csv"):
        topic, virtue = r["topic"].strip().lower(), r["virtue"].strip().lower()
        fit = float(r["fit"])
        if virtue not in virtues:
            raise ValueError(f"virtue_fit.csv: unknown virtue {virtue!r}")
        if not 0.0 <= fit <= 1.0:
            raise ValueError(f"virtue_fit.csv: fit {fit} for {topic}/{virtue} is outside 0 to 1")
        if topic not in topics:
            topics.append(topic)
        pairs[(topic, virtue)] = Pair(topic, virtue, fit, r["title"].strip(), r["why"].strip())

    activities: Dict[Tuple[str, str, str], Activity] = {}
    for r in _rows(folder / "activities.csv"):
        topic, virtue, band = r["topic"].strip().lower(), r["virtue"].strip().lower(), r["band"].strip()
        activities[(topic, virtue, band)] = Activity(
            topic, virtue, band, int(r["minutes"]), r["activity"].strip())

    missing = [
        (t, v, b) for t in topics for v in virtues for b in BANDS
        if (t, v) not in pairs or (t, v, b) not in activities
    ]
    if missing:
        raise ValueError(f"catalog is incomplete, for example {missing[0]} (+{len(missing) - 1} more)")
    return Catalog(topics, virtues, pairs, activities)
