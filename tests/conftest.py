"""A tiny catalog (2 topics, 4 virtues, 4 grade bands) for fast, predictable tests."""

import csv

import pytest

BANDS = ["K-2", "3-5", "6-8", "9-12"]
VIRTUES = {
    "perseverance": ("Trying again", "Perseverance is keeping going when something is hard.", "frustrated|giving-up|stuck"),
    "teamwork": ("Working together", "Teamwork is working with others toward a shared goal.", "together|partners|bossy"),
    "justice": ("Fairness", "Justice means treating people fairly with fair turns.", "unfair|sharing|turns"),
    "honesty": ("Telling the truth", "Honesty is telling the truth about what we did.", "lying|cheating|blaming"),
}
# topic -> virtue -> (fit, title, why, minutes per band)
FIT = {
    "loops": {
        "perseverance": (1.0, "Cat Keeps Trying", "A loop repeats a small effort until the goal is reached.", [10, 15, 20, 30]),
        "teamwork": (0.8, "Dance Loop", "A repeating dance made by two people needs them in step.", [10, 15, 20, 30]),
        "justice": (0.5, "Everyone Gets a Turn", "A loop that visits every child once models fair turns.", [10, 15, 20, 30]),
        "honesty": (0.4, "Count It Truly", "Counting how often a loop really ran is honest reporting.", [5, 10, 15, 20]),
    },
    "events": {
        "perseverance": (0.4, "Try the Button", "Trying an event again until it fires builds patience.", [10, 15, 20, 30]),
        "teamwork": (0.6, "You Send I Respond", "One part sends an event and another responds.", [10, 15, 20, 30]),
        "justice": (0.5, "Same Event Same Response", "Programs answer the same event the same way for everyone.", [10, 15, 20, 30]),
        "honesty": (0.4, "Does It Respond", "Checking that an event really fires is honest testing.", [10, 15, 20, 30]),
    },
}


@pytest.fixture
def data_dir(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    with open(folder / "virtues.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["virtue", "meaning", "guide", "cue_tags"])
        for v, (meaning, guide, tags) in VIRTUES.items():
            w.writerow([v, meaning, guide, tags])
    with open(folder / "virtue_fit.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["topic", "virtue", "fit", "title", "why"])
        for t, vs in FIT.items():
            for v, (fit, title, why, _) in vs.items():
                w.writerow([t, v, fit, title, why])
    with open(folder / "activities.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["topic", "virtue", "band", "minutes", "activity"])
        for t, vs in FIT.items():
            for v, (_, title, _, mins) in vs.items():
                for band, m in zip(BANDS, mins):
                    w.writerow([t, v, band, m, f"{title} activity for {band}."])
    return str(folder)
