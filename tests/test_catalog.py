import pytest

from src.catalog import BANDS, GRADES, grade_to_band, load_catalog


@pytest.mark.parametrize("grade,band", [
    ("K", "K-2"), ("k", "K-2"), (1, "K-2"), ("2", "K-2"), ("3", "3-5"), (5, "3-5"),
    ("6", "6-8"), (8, "6-8"), ("9", "9-12"), (12, "9-12"),
])
def test_grade_to_band(grade, band):
    assert grade_to_band(grade) == band


@pytest.mark.parametrize("grade", ["0", 13, "", "pre-k", None, "1.5"])
def test_bad_grades_are_rejected(grade):
    with pytest.raises(ValueError):
        grade_to_band(grade)


def test_every_listed_grade_has_a_band():
    assert {grade_to_band(g) for g in GRADES} == set(BANDS)


def test_the_real_catalog_is_complete():
    cat = load_catalog("data")
    assert len(cat.topics) == 12 and len(cat.virtues) == 8
    assert len(cat.pairs) == 96 and len(cat.activities) == 384
    assert all(0.0 <= p.fit <= 1.0 and p.title and p.why for p in cat.pairs.values())
    assert all(a.minutes > 0 and a.text for a in cat.activities.values())


def test_small_catalog_loads(data_dir):
    cat = load_catalog(data_dir)
    assert cat.topics == ["loops", "events"]
    assert cat.activity("loops", "perseverance", "3-5").minutes == 15


def test_missing_activity_is_an_error(data_dir):
    path = f"{data_dir}/activities.csv"
    lines = open(path, encoding="utf-8").read().splitlines()
    open(path, "w", encoding="utf-8").write("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="incomplete"):
        load_catalog(data_dir)


def test_fit_outside_range_is_an_error(data_dir):
    path = f"{data_dir}/virtue_fit.csv"
    text = open(path, encoding="utf-8").read().replace(",1.0,", ",1.5,", 1)
    open(path, "w", encoding="utf-8").write(text)
    with pytest.raises(ValueError, match="outside 0 to 1"):
        load_catalog(data_dir)
