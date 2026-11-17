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

from src.catalog import GRADES, load_catalog
from src.pipeline import RunResult, fit_text, run_pipeline

BASE = Path(__file__).parent
STATIC = BASE / "static"
DATA = BASE / "data"
TRACE_DIR = BASE / "traces"

log = logging.getLogger("virtueweave")
app = FastAPI(title="VirtueWeave")


class Ask(BaseModel):
    topic: str = Field(max_length=80)
    grade: str = Field(max_length=10)
    minutes: int
    note: str = Field(default="", max_length=2000)


class Feedback(BaseModel):
    trace_id: str = Field(pattern=r"^run-\d{8}-\d{6}-[0-9a-f]{8}$")
    virtue: str = Field(pattern=r"^[a-z]{3,20}$")
    decision: Literal["accept", "reject"]


def serialize(result: RunResult) -> Dict[str, Any]:
    """Turns a RunResult into the JSON the page draws."""
    trace_id = Path(result.trace_path).stem if result.trace_path else None
    r = result.request
    return {
        "ok": result.ok and bool(result.picks),
        "message": result.message,
        "trace_id": trace_id,
        "request": None if r is None else {
            "topic": r.topic, "grade": r.grade, "band": r.band,
            "minutes": r.minutes, "note": r.note,
        },
        "explained": result.explained,
        "virtues": [{
            "virtue": p.virtue,
            "fits": fit_text(p, result),
            "written_by_model": p.virtue in result.virtue_explanations,
            "title": p.title,
            "activity": p.activity,
            "minutes": p.activity_minutes,
            "over_time": p.over_time,
        } for p in result.picks],
    }


@app.post("/api/recommend")
def recommend(body: Ask):
    try:
        result = run_pipeline(
            body.topic, body.grade, body.minutes, body.note,
            data_dir=str(DATA), trace_dir=str(TRACE_DIR),
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


@app.get("/api/catalog")
def catalog():
    """What the form offers and the counts shown on the home page."""
    cat = load_catalog(str(DATA))
    return {
        "topics": cat.topics,
        "grades": GRADES,
        "virtues": {name: info.meaning for name, info in cat.virtues.items()},
        "pairs": len(cat.pairs),
        "activities": len(cat.activities),
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
