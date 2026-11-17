import json

import pytest
from fastapi.testclient import TestClient

import server
from src.llm import LLMUnavailable

BODY = {"topic": "loops", "grade": "1", "minutes": 15, "note": "they keep giving up"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "TRACE_DIR", tmp_path / "traces")
    real = server.run_pipeline

    def offline(*args, **kwargs):          # never touch the network in tests
        def down(system, user, schema):
            raise LLMUnavailable("test")
        return real(*args, **kwargs, model_call=down)

    monkeypatch.setattr(server, "run_pipeline", offline)
    return TestClient(server.app)


def test_home_page_and_static_files_are_served(client):
    assert "VirtueWeave" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_catalog_lists_what_the_form_offers(client):
    body = client.get("/api/catalog").json()
    assert len(body["topics"]) == 12 and "loops" in body["topics"]
    assert body["grades"][0] == "K" and body["grades"][-1] == "12"
    assert len(body["virtues"]) == 8 and body["pairs"] == 96 and body["activities"] == 384


def test_recommend_returns_three_virtues_with_activities(client):
    body = client.post("/api/recommend", json=BODY).json()
    assert body["ok"] and body["request"]["band"] == "K-2"
    assert len(body["virtues"]) == 3 and body["virtues"][0]["virtue"] == "perseverance"
    first = body["virtues"][0]
    assert first["fits"] and first["title"] and first["activity"] and first["minutes"] == 15
    assert not {"reasons", "sources"} & set(first) and "steps" not in body
    assert first["written_by_model"] is False        # model was offline, catalog text shown
    assert body["trace_id"].startswith("run-")


def test_blocked_input_gives_a_message_not_an_error(client):
    body = client.post("/api/recommend", json={**BODY, "topic": "nope"}).json()
    assert body["ok"] is False and body["message"] and body["virtues"] == []


@pytest.mark.parametrize("payload", [
    {**BODY, "minutes": "soon"},
    {**BODY, "note": "x" * 2001},
    {"topic": "loops"},
])
def test_malformed_requests_are_rejected(client, payload):
    assert client.post("/api/recommend", json=payload).status_code == 422


def test_feedback_is_recorded(client):
    trace_id = client.post("/api/recommend", json=BODY).json()["trace_id"]
    ok = client.post("/api/feedback", json={"trace_id": trace_id, "virtue": "perseverance", "decision": "accept"})
    assert ok.status_code == 200
    lines = (server.TRACE_DIR / "feedback.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["decision"] == "accept" and json.loads(lines[0])["virtue"] == "perseverance"


@pytest.mark.parametrize("payload", [
    {"trace_id": "../../etc/passwd", "virtue": "justice", "decision": "accept"},
    {"trace_id": "run-20260101-000000-abcdef12", "virtue": "justice", "decision": "maybe"},
    {"trace_id": "run-20260101-000000-abcdef12", "virtue": "Not A Virtue!", "decision": "accept"},
])
def test_bad_feedback_is_rejected(client, payload):
    assert client.post("/api/feedback", json=payload).status_code == 422
