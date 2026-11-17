from types import SimpleNamespace

from src.evaluate import CASES, Case, run_evaluation, violations


def fake_result(picks, expect="answer"):
    return SimpleNamespace(picks=[
        SimpleNamespace(practice={"id": i, "title": f"P{i}", "age_min": lo, "age_max": hi, "minutes": m})
        for i, (lo, hi, m) in enumerate(picks, start=1)
    ])


def test_case_mix_is_5_ordinary_2_vague_5_guardrail():
    kinds = [c.kind for c in CASES]
    assert (kinds.count("ordinary"), kinds.count("vague"), kinds.count("guardrail")) == (5, 2, 5)
    assert all(c.expect_virtue for c in CASES if c.kind == "ordinary")


def test_violations_catch_age_minutes_count_and_duplicates():
    case = Case("x", "ordinary", "s", "justice", age=6, minutes=10)
    assert violations(case, fake_result([(5, 7, 10)])) == []
    found = violations(case, fake_result([(9, 11, 10), (5, 7, 15)]))
    assert any("exclude 6" in v for v in found) and any("over 10" in v for v in found)
    assert "4 picks (max 3)" in violations(case, fake_result([(5, 7, 5)] * 4))[0]
    blocked = Case("b", "guardrail", "s", expect="blocked")
    assert violations(blocked, fake_result([(5, 7, 5)])) [0] == "picks returned after a block"


def test_offline_evaluation_runs_all_12_and_guardrails_hold(tmp_path):
    report = run_evaluation(live=False, repeats=2, out_dir=str(tmp_path))
    assert len(report["cases"]) == 12
    guardrails = [r for r in report["cases"] if r["kind"] == "guardrail"]
    assert all(r["passed"] for r in guardrails)
    assert report["summary"]["constraint_violations"] == 0
    assert report["summary"]["same_result_on_repeat_runs"] == "12/12"
    assert all(r["trace"] is None for r in report["cases"])   # scratch traces not kept
