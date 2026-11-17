"""
VirtueWeave web server: a small API around run_pipeline() plus the static site.

    python -m uvicorn server:app --port 8000

then open http://localhost:8000
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.explainer import group_by_virtue
from src.pipeline import RunResult, run_pipeline

BASE = Path(__file__).parent
STATIC = BASE / "static"
DATA = BASE / "data"
TRACE_DIR = BASE / "traces"

log = logging.getLogger("virtueweave")
app = FastAPI(title="VirtueWeave")


class Ask(BaseModel):
    sentence: str = Field(max_length=2000)


class Feedback(BaseModel):
    trace_id: str = Field(pattern=r"^run-\d{8}-\d{6}-[0-9a-f]{8}$")
    practice_id: int
    decision: Literal["accept", "reject"]


def serialize(result: RunResult) -> Dict[str, Any]:
    """Turns a RunResult into the JSON the page draws."""
    trace_id = Path(result.trace_path).stem if result.trace_path else None
    steps = []
    if result.trace_path:
        with open(result.trace_path, encoding="utf-8") as f:
            steps = [{"step": s["step"], "status": s["status"], "ms": s.get("ms")}
                     for s in json.load(f)["steps"]]
    i = result.interpretation
    virtues = []
    for virtue, group in group_by_virtue(result.picks).items():
        virtues.append({
            "virtue": virtue,
            "explanation": result.virtue_explanations.get(virtue),
            "practices": [{
                "id": p.practice["id"], "title": p.practice["title"],
                "topic": p.practice["topic"], "minutes": p.practice["minutes"],
                "age_min": p.practice["age_min"], "age_max": p.practice["age_max"],
                "reasons": p.reasons, "sources": p.evidence,
            } for p in group],
        })
    return {
        "ok": result.ok and bool(result.picks),
        "message": result.message,
        "trace_id": trace_id,
        "understood": None if i is None else {
            "topic": i.topic, "age": i.age, "minutes": i.minutes,
            "description": i.description, "source": i.source,
        },
        "explained": result.explained,
        "virtues": virtues,
        "steps": steps,
    }


@app.post("/api/ask")
def ask(body: Ask):
    try:
        result = run_pipeline(
            body.sentence,
            practices_path=str(DATA / "practices.csv"),
            notes_path=str(DATA / "notes.csv"),
            trace_dir=str(TRACE_DIR),
        )
        return serialize(result)
    except Exception:
        log.exception("pipeline failed")
        return JSONResponse({"error": "Something went wrong on our side."}, status_code=500)


@app.post("/api/feedback")
def feedback(body: Feedback):
    """Records whether the teacher accepted or rejected a practice."""
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    line = {"when": datetime.now(timezone.utc).isoformat(), **body.model_dump()}
    with open(TRACE_DIR / "feedback.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
