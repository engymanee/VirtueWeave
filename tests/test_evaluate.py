from src.catalog import load_catalog
from src.evaluate import CASES, Case, run_evaluation, violations
from src.pipeline import Pick, RunResult


def result_with(virtues, catalog, case):
    from src.catalog import grade_to_band
    band = grade_to_band(case.grade)
    picks = []
    for v in virtues:
        act = catalog.activity(case.topic, v, band)
        picks.append(Pick(v, 1.0, "t", "w", act.text, act.minutes,
                          act.minutes > int(case.minutes), [], []))
    return RunResult(ok=True, picks=picks)


def test_case_mix_is_5_ordinary_2_vague_5_guardrail():
    kinds = [c.kind for c in CASES]
    assert (kinds.count("ordinary"), kinds.count("vague"), kinds.count("guardrail")) == (5, 2, 5)
    assert all(c.expect_virtue for c in CASES if c.kind == "ordinary")


def test_violations_pass_a_clean_result_and_catch_problems():
    cat = load_catalog("data")
    case = Case("x", "ordinary", "loops", "1", 20, "", "perseverance")
    assert violations(case, result_with(["perseverance", "teamwork", "justice"], cat, case), cat) == []
    assert any("3 distinct" in v for v in violations(case, result_with(["justice", "justice", "teamwork"], cat, case), cat))

    wrong_band = result_with(["perseverance", "teamwork", "justice"], cat, case)
    wrong_band.picks[0].activity = "something else"
    assert any("not the K-2 one" in v for v in violations(case, wrong_band, cat))

    bad_flag = result_with(["perseverance", "teamwork", "justice"], cat, case)
    bad_flag.picks[0].over_time = not bad_flag.picks[0].over_time
    assert any("over-time flag" in v for v in violations(case, bad_flag, cat))

    blocked = Case("b", "guardrail", "loops", "99", 5, "", expect="blocked")
    assert violations(blocked, result_with(["justice"], cat, case), cat) == ["picks returned after a block"]


def test_offline_evaluation_runs_all_12_and_the_basics_hold(tmp_path):
    report = run_evaluation(live=False, repeats=2, out_dir=str(tmp_path))
    assert len(report["cases"]) == 12
    assert all(r["passed"] for r in report["cases"] if r["kind"] == "guardrail")
    assert report["summary"]["constraint_violations"] == 0
    assert report["summary"]["same_result_on_repeat_runs"] == "12/12"
