"""
Runs the agent on three example classes with the real model and saves what it did.

    python3 -m src.agent_demo

Needs an API key (about 3 short tool-use runs). For each request it saves the
agent's full trace to assets/agent_traces/ and writes a readable version into
ai_interactions.md. It stops without writing anything if the agent cannot be
reached, so the log only ever contains real runs.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from src.docs_section import replace_section
from src.llm import unavailable
from src.pipeline import run_pipeline

REQUESTS = [
    {"topic": "loops", "grade": "1", "minutes": 15, "note": "they keep giving up"},
    {"topic": "debugging", "grade": "9", "minutes": 30, "note": "some blame their partner when it breaks"},
    {"topic": "patterns", "grade": "K", "minutes": 5, "note": ""},
]


def run_demo(
    requests: List[Dict[str, Any]] = REQUESTS,
    tool_call: Any = None,
    out_dir: str = "assets/agent_traces",
    doc_path: str = "ai_interactions.md",
    data_dir: str = "data",
    trace_dir: str = "traces",
) -> List[Dict[str, Any]]:
    """Runs the agent for each request. Returns the saved runs."""
    extra = {} if tool_call is None else {"tool_call": tool_call}
    folder = Path(out_dir)
    saved = []
    for number, req in enumerate(requests, start=1):
        result = run_pipeline(req["topic"], req["grade"], req["minutes"], req["note"],
                              data_dir=data_dir, trace_dir=trace_dir,
                              model_call=unavailable, agent=True, **extra)
        if not (result.agent and result.agent["used"]):
            reason = result.agent["fallback_reason"] if result.agent else "agent mode was not run"
            raise SystemExit(f"Stopped: the agent did not run ({reason}). Nothing was written.")
        trace = json.load(open(result.trace_path, encoding="utf-8"))
        agent_step = next(s for s in trace["steps"] if s["step"] == "agent")
        slug = re.sub(r"[^a-z0-9]+", "-", req["topic"].lower()).strip("-")
        folder.mkdir(parents=True, exist_ok=True)
        file = folder / f"agent-{number}-{slug}.json"
        record = {"request": req, **{k: agent_step[k] for k in
                  ("steps", "reasoning", "scorer_top3", "agent_top3", "overlap")}}
        file.write_text(json.dumps(record, indent=2), encoding="utf-8")
        saved.append({**record, "file": file.as_posix()})
    replace_section(doc_path, "agent-traces", render_markdown(saved))
    return saved


def _cell(text: str, limit: int = 110) -> str:
    text = " ".join(str(text).split()).replace("|", "/")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def render_markdown(runs: List[Dict[str, Any]]) -> str:
    out = [
        "## Project 4: agent workflow traces",
        "",
        "In agent mode (`python3 -m src.main ... --agent`) the model chooses the three virtues "
        "by calling tools: `get_topic_fits`, `search_class_notes`, `get_activity` and "
        "`submit_recommendation`. Every tool call carries the agent's stated reason. The traces "
        "below are real runs, saved in full as JSON in `assets/agent_traces/`. The code is in "
        "`src/agent.py`, and the pipeline falls back to the scorer if the agent cannot answer.",
    ]
    for n, run in enumerate(runs, start=1):
        r = run["request"]
        note = f', note: "{r["note"]}"' if r["note"] else ", no note"
        out += ["", f"### Run {n}: {r['topic']}, grade {r['grade']}, {r['minutes']} min{note}", "",
                "| Step | Tool | Agent's reason | Result |", "|---|---|---|---|"]
        for s in run["steps"]:
            out.append(f"| {s['step']} | `{s['tool']}` | {_cell(s['reason'])} | {_cell(s['result'])} |")
        out += ["",
                f"- **Agent's choice:** {', '.join(run['agent_top3'])}",
                f"- **Agent's reasoning:** {run['reasoning']}",
                f"- **Scorer's choice:** {', '.join(run['scorer_top3'])} "
                f"(the agent agrees on {run['overlap']} of 3)",
                f"- **Full trace:** [{run['file']}]({run['file']})"]
    return "\n".join(out)


if __name__ == "__main__":
    runs = run_demo()
    print(f"Saved {len(runs)} agent traces to assets/agent_traces/ and updated ai_interactions.md")
