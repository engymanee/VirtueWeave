import pytest

from src.catalog import load_catalog
from src.retrieval import (
    CUE_BONUS, MIN_SIMILARITY, VirtueRetriever, passages_from_catalog, stem,
)


@pytest.fixture
def retriever(data_dir):
    return VirtueRetriever.from_catalog(load_catalog(data_dir))


def test_passages_cover_guides_whys_and_activities(data_dir):
    passages = passages_from_catalog(load_catalog(data_dir))
    by_source = {s: [p for p in passages if p.source == s] for s in ("guide", "why", "activity")}
    assert (len(by_source["guide"]), len(by_source["why"]), len(by_source["activity"])) == (4, 8, 32)
    assert all(p.band for p in by_source["activity"]) and all(p.topic for p in by_source["why"])


def test_stemmer_joins_word_forms():
    assert stem("teasing") == stem("tease") == stem("teases") == stem("teased")
    assert stem("blame") == stem("blaming")
    assert stem("running") == stem("run")
    assert stem("is") == "is" and stem("go") == "go"


def test_note_matches_the_virtue_it_is_about(retriever):
    matches = retriever.match_virtues("they need fair turns to take part")
    best = max(matches.values(), key=lambda m: m.score)
    assert best.virtue == "justice" and best.method.startswith("tfidf")


def test_scores_are_between_zero_and_one(retriever):
    for match in retriever.match_virtues("fair turns and a goal reached together").values():
        assert 0.0 <= match.score <= 1.0


def test_evidence_belongs_to_its_virtue_and_is_sorted(retriever):
    match = retriever.match_virtues("repeats a small effort until the goal is reached")["perseverance"]
    assert match.evidence and all(h.passage.virtue == "perseverance" for h in match.evidence)
    scores = [h.score for h in match.evidence]
    assert scores == sorted(scores, reverse=True)


def test_evidence_per_virtue_is_capped(retriever):
    matches = retriever.match_virtues("fair turns every child goal", evidence_per_virtue=1)
    assert all(len(m.evidence) <= 1 for m in matches.values())


def test_word_forms_match_across_note_and_text(retriever):
    assert any(h.score >= MIN_SIMILARITY for h in retriever.search("fairly takes turn"))


def test_cue_tag_adds_a_bonus_on_top_of_the_text_match(retriever):
    plain = retriever.match_virtues("treating people fairly")["justice"]
    tagged = retriever.match_virtues("treating people fairly, it feels unfair")["justice"]
    assert tagged.cue_hits == ["unfair"] and tagged.method == "tfidf+cue-tags"
    assert tagged.score > plain.score
    assert tagged.score <= min(1.0, plain.score + CUE_BONUS + 0.3)
    assert plain.cue_hits == []


def test_tag_forms_match_the_note(retriever):
    assert retriever.match_virtues("he gives up quickly")["perseverance"].cue_hits == ["giving-up"]


def test_weak_text_matches_are_ignored(retriever, monkeypatch):
    import src.retrieval as retrieval
    note = "repeats a small effort until the goal is reached"
    assert retriever.match_virtues(note)["perseverance"].method == "tfidf"
    monkeypatch.setattr(retrieval, "MIN_SIMILARITY", 0.99)   # nothing is strong enough now
    matches = retriever.match_virtues(note)
    assert all(m.method == "cue-tags" and m.score == 0.0 for m in matches.values())


def test_no_text_match_falls_back_to_the_share_of_cue_tags(retriever):
    matches = retriever.match_virtues("so frustrated and stuck")
    assert matches["perseverance"].method == "cue-tags"
    assert matches["perseverance"].score == pytest.approx(2 / 3)
    assert matches["teamwork"].score == 0.0


def test_empty_note_scores_zero_everywhere(retriever):
    assert retriever.search("   ") == []
    assert all(m.score == 0.0 for m in retriever.match_virtues("").values())


def test_empty_index_is_an_error():
    with pytest.raises(ValueError):
        VirtueRetriever([])
