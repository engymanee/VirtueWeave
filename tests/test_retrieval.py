import pytest

from src.retrieval import (
    Passage, VirtueRetriever, format_evidence, load_cue_tags, load_guide_passages,
    load_note_passages,
)

PRACTICES = (
    "id,title,virtue,topic,fit,minutes,age_min,age_max,cue_tags\n"
    "1,Turns,justice,sequencing,1.0,10,5,7,unfair|sharing|turns|left-out\n"
    "2,Retry,perseverance,debugging,1.0,10,5,7,frustrated|giving-up|stuck\n"
)
NOTES = (
    "id,why,how\n"
    '1,Building a sequence one block at a time gives each child an equal share.,'
    '"Pairs build a program, adding one block each in turn."\n'
    '2,Finding the step that fails means trying again with a plan.,'
    '"Children find the broken block, fix it and run again."\n'
)
GUIDES = (
    "virtue,text,title\n"
    "justice,Fairness means everyone gets an equal chance to take part.,Justice basics\n"
    "perseverance,Perseverance is carrying on after a setback.,Perseverance basics\n"
)


@pytest.fixture
def files(tmp_path):
    paths = {}
    for name, content in (("practices", PRACTICES), ("notes", NOTES), ("guides", GUIDES)):
        path = tmp_path / f"{name}.csv"
        path.write_text(content, encoding="utf-8")
        paths[name] = str(path)
    return paths


@pytest.fixture
def retriever(files):
    return VirtueRetriever.from_files(files["notes"], files["practices"], files["guides"])


def test_note_passages_split_why_and_how_with_virtue(files):
    passages = load_note_passages(files["notes"], files["practices"])
    assert len(passages) == 4
    assert {p.virtue for p in passages} == {"justice", "perseverance"}
    assert {p.label for p in passages if p.practice_id == 1} == {"Turns (why)", "Turns (how)"}


def test_note_with_unknown_practice_id_is_an_error(tmp_path, files):
    bad = tmp_path / "bad_notes.csv"
    bad.write_text("id,why,how\n99,x,y\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_note_passages(str(bad), files["practices"])


def test_guide_passages_load_with_titles(files):
    passages = load_guide_passages(files["guides"])
    assert [p.label for p in passages] == ["Justice basics", "Perseverance basics"]
    assert all(p.source == "guide" for p in passages)


def test_cue_tags_are_collected_per_virtue(files):
    tags = load_cue_tags(files["practices"])
    assert "left-out" in tags["justice"]
    assert "stuck" in tags["perseverance"]


def test_matching_description_scores_its_virtue_highest(retriever):
    matches = retriever.match_virtues("the children need an equal share and a turn each")
    assert matches["justice"].score > matches["perseverance"].score
    assert matches["justice"].method.startswith("tfidf")


def test_scores_are_between_zero_and_one(retriever):
    for match in retriever.match_virtues("fix the block and run again").values():
        assert 0.0 <= match.score <= 1.0


def test_evidence_comes_from_the_matching_virtue_and_is_sorted(retriever):
    match = retriever.match_virtues("run the program again to fix the block")["perseverance"]
    assert match.evidence
    assert all(h.passage.virtue == "perseverance" for h in match.evidence)
    scores = [h.score for h in match.evidence]
    assert scores == sorted(scores, reverse=True)


def test_evidence_includes_guides_as_well_as_notes(retriever):
    hits = retriever.search("carrying on after a setback")
    assert any(h.passage.source == "guide" for h in hits)


def test_evidence_per_virtue_is_capped(retriever):
    matches = retriever.match_virtues("equal share each child block turn", evidence_per_virtue=1)
    assert all(len(m.evidence) <= 1 for m in matches.values())


def test_no_text_match_falls_back_to_cue_tags(retriever):
    matches = retriever.match_virtues("She feels left out and it is unfair")
    assert matches["justice"].method == "cue-tags"
    assert matches["justice"].cue_hits == ["unfair", "left-out"]
    assert matches["justice"].score == pytest.approx(2 / 4)
    assert matches["perseverance"].score == 0.0
    assert matches["justice"].evidence == []


def test_hyphenated_tag_matches_spaced_phrase(retriever):
    matches = retriever.match_virtues("he keeps giving up")
    assert matches["perseverance"].cue_hits == ["giving-up"]


def test_empty_description_scores_zero_everywhere(retriever):
    assert retriever.search("   ") == []
    assert all(m.score == 0.0 for m in retriever.match_virtues("").values())


def test_text_match_wins_over_cue_tags_when_both_exist(retriever):
    matches = retriever.match_virtues("everyone is stuck and needs an equal share")
    assert all(m.method.startswith("tfidf") for m in matches.values())


def test_empty_index_is_an_error():
    with pytest.raises(ValueError):
        VirtueRetriever([])


def test_format_evidence_names_source_and_text():
    passage = Passage("guide", "justice", "Fairness matters.", "Justice basics")
    from src.retrieval import Hit, VirtueMatch
    match = VirtueMatch("justice", 0.5, "tfidf", [Hit(passage, 0.5)], [])
    line = format_evidence(match)
    assert "[guide: Justice basics]" in line and "Fairness matters." in line


def test_stemmer_joins_word_forms():
    from src.retrieval import stem
    assert stem("teasing") == stem("tease") == stem("teases") == stem("teased")
    assert stem("blame") == stem("blaming")
    assert stem("running") == stem("run")
    assert stem("is") == "is" and stem("go") == "go"          # short words are left alone


def test_text_search_matches_other_forms_of_a_word(retriever):
    hits = retriever.search("they are sharing")   # the note says "share"
    assert hits and hits[0].passage.virtue == "justice"


def test_cue_tag_adds_a_bonus_on_top_of_text_matches(retriever):
    from src.retrieval import CUE_BONUS
    plain = retriever.match_virtues("an equal chance each")["justice"]
    tagged = retriever.match_virtues("an equal chance each, it feels unfair")["justice"]
    assert tagged.cue_hits == ["unfair"] and tagged.method == "tfidf+cue-tags"
    assert tagged.score == pytest.approx(min(1.0, plain.score + CUE_BONUS), abs=0.05)
    assert tagged.score > plain.score


def test_tag_forms_match_the_description(retriever):
    # tag is "giving-up"; the teacher writes "gives up"
    assert retriever.match_virtues("he gives up quickly")["perseverance"].cue_hits == ["giving-up"]
