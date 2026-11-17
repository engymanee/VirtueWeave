import json

import pytest
from fastapi.testclient import TestClient

import server

SENTENCE = "Teaching loops to my 6-year-olds tomorrow, 15 minutes, they keep giving up."


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "TRACE_DIR", tmp_path / "traces")
    # Make every model call fail so the test never touches the network.
    import src.llm
    def down(*a, **k):
        raise src.llm.LLMUnavailable("test")
    monkeypatch.setattr("src.pipeline.call_json", down, raising=False)
    monkeypatch.setattr(server, "run_pipeline",
                        lambda *a, **k: _run(*a, **k, model_call=down))
    return TestClient(server.app)


_real_run = server.run_pipeline


def _run(*args, **kwargs):
    return _real_run(*args, **kwargs)


def test_home_page_and_static_files_are_served(client):
    assert "VirtueWeave" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_ask_returns_virtue_cards_with_practices_and_steps(client):
    body = client.post("/api/ask", json={"sentence": SENTENCE}).json()
    assert body["ok"] and body["understood"]["topic"] == "loops"
    assert body["virtues"][0]["virtue"] == "perseverance"
    practice = body["virtues"][0]["practices"][0]
    assert practice["reasons"] and practice["sources"]
    assert [s["step"] for s in body["steps"]][0] == "guard_input"
    assert body["trace_id"].startswith("run-")


def test_blocked_sentence_gives_a_message_not_an_error(client):
    body = client.post("/api/ask", json={"sentence": "   "}).json()
    assert body["ok"] is False and body["message"] and body["virtues"] == []


def test_overlong_body_is_rejected(client):
    assert client.post("/api/ask", json={"sentence": "x" * 2001}).status_code == 422


def test_feedback_is_recorded(client):
    trace_id = client.post("/api/ask", json={"sentence": SENTENCE}).json()["trace_id"]
    ok = client.post("/api/feedback",
                     json={"trace_id": trace_id, "practice_id": 7, "decision": "accept"})
    assert ok.status_code == 200
    lines = (server.TRACE_DIR / "feedback.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["decision"] == "accept"


@pytest.mark.parametrize("payload", [
    {"trace_id": "../../etc/passwd", "practice_id": 1, "decision": "accept"},
    {"trace_id": "run-20260101-000000-abcdef12", "practice_id": 1, "decision": "maybe"},
])
def test_bad_feedback_is_rejected(client, payload):
    assert client.post("/api/feedback", json=payload).status_code == 422
