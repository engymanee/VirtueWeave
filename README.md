# VirtueWeave

**A virtue recommender system for computer science teachers.**

Tell VirtueWeave the topic you are teaching, the grade, and how many minutes you have. It recommends the **three virtues** that fit best. For each one it explains **why it fits** the topic and **how to bring it into the lesson**, with an activity sized to the grade and the time.

## Why it matters

Teachers are asked to build character as well as skills, but virtue education is rarely connected to the lessons they already teach. A teacher planning a lesson on loops does not have time to work out how perseverance, fairness or gratitude could live inside it. VirtueWeave does that matching for them in seconds and shows its reasoning, so the teacher can trust it, adapt it, or ignore it.

## Where it came from

VirtueWeave grew out of the **Music Recommender Simulation**. That project scored songs against a listener's taste profile (genre, mood, energy, and later popularity, decade, mood tags, language and instrumentalness). It had switchable ranking strategies, a diversity penalty for repeated artists and genres, and a table of reasons that explained every score. VirtueWeave keeps the same core idea: score the options, rank them, and say why. It applies that idea to virtues and lessons instead of songs, and it adds retrieval, a language model, guardrails and a web interface.

## What you get

| You enter | You get |
|---|---|
| A topic (12 to choose from), a grade (K-12), minutes, and an optional note such as "they keep giving up" | Three virtues, each with **Why it fits** (how the virtue connects to the topic) and **How to bring it in** (a titled activity for that grade band, with its length) |

## Architecture overview

The system is a short pipeline over three small data files. One function, `run_pipeline()` in `src/pipeline.py`, runs the steps in order and writes a JSON trace of every run to `traces/`.

```mermaid
flowchart TD
    A["Teacher chooses a topic, a grade (K-12) and minutes<br/>and may add a note about the class"] --> B{"Input guardrail<br/>topic is one of 12, grade K-12,<br/>minutes 1-180, safe note"}
    B -- blocked --> X["Stop with a clear message"]
    B -- ok --> C{"Is there a note?"}

    C -- yes --> R["Retrieve: TF-IDF over virtue guides,<br/>why-it-fits notes and activities<br/>plus a bonus for cue tags"]
    C -- no --> K["Score the 8 virtues<br/>for the topic"]
    R --> K

    G[("virtues.csv<br/>8 guides and cue tags")] --> R
    W[("virtue_fit.csv<br/>96 topic x virtue pairs:<br/>fit and why it fits")] --> R
    W --> K
    AC[("activities.csv<br/>384 activities, one per<br/>grade band: K-2, 3-5, 6-8, 9-12")] --> R
    AC --> K

    K --> T3["Pick the top 3 virtues<br/>score = fit + note match<br/>activities that fit the time come first,<br/>longer ones are flagged"]
    T3 --> M{"Agent mode on?"}
    M -- no --> E
    M -- yes --> AG["Agent: the model chooses the 3 virtues by calling tools<br/>topic fits, note search, activity check<br/>each step and reason is logged"]
    AG -- unavailable or invalid --> E
    AG --> E
    E["Explain: the model writes 2-3 sentences<br/>per virtue on how it fits the topic"]
    E -- model unavailable --> H["Use the catalog's why-it-fits text"]
    E --> F{"Output guardrail<br/>right virtues, 2-3 sentences,<br/>short, no links"}
    F -- fail --> H
    F -- pass --> I["Three virtue cards<br/>Why it fits<br/>How to bring it in"]
    H --> I
    I --> J["Teacher accepts or rejects each virtue"]
    J --> FB[("feedback.jsonl")]
    I --> TR[("Trace log: one JSON file per run")]
```

1. **Input guardrail.** Checks that the topic is one of the 12, the grade is K-12, the minutes are 1-180, and the note is short and does not try to give the assistant instructions. Anything else stops with a clear message.
2. **Retrieve (only if there is a note).** A TF-IDF index (scikit-learn) covers the virtue guides, the "why it fits" notes and the activities. The note is the query. Each virtue gets a score from its best-matching passage, plus a bonus when one of its cue tags (for example "teasing" or "blaming") appears in the note. Weak matches are ignored.
3. **Rank.** Every one of the 8 virtues is scored: its fit for the topic from the table, plus the note match. Virtues whose activity fits the teacher's time come first; any that run longer are flagged. The top 3 are kept.
4. **Explain.** One language-model call writes two or three warm sentences per virtue about how it fits the topic. The model is given the catalog's own text and the scorer's reasons, so it rewords facts it was handed and does not invent them. If the model is unavailable, the catalog's own "why it fits" text is shown.
5. **Output guardrail.** Checks the model's wording: the right virtues, 2-3 sentences each, a length limit, no links. If it fails, the catalog text is shown instead.

**Agent mode (optional).** With `--agent`, step 3 is done by the model instead of the formula. The agent works in steps by calling tools (`get_topic_fits`, `search_class_notes`, `get_activity`, then `submit_recommendation`), and every call carries its stated reason. Its choice is checked (exactly 3 distinct, real virtues) and compared with the scorer's, and if the model is unavailable or never submits a valid answer the scorer's choice is used. Every step is saved in the run's trace. The code is in `src/agent.py`.

**The data** (`data/`):

| File | Contents |
|---|---|
| `virtues.csv` | 8 virtues, each with a short guide and cue tags |
| `virtue_fit.csv` | 96 rows, one for every topic x virtue pair: a fit score (0-1), an activity title and a "why it fits" sentence |
| `activities.csv` | 384 rows: one activity for every pair in each of four grade bands (K-2, 3-5, 6-8, 9-12), with its length in minutes |

**The code** (`src/`): `catalog.py` loads and validates the data, `retrieval.py` is the TF-IDF index, `pipeline.py` is the pipeline, `explainer.py` is the model step and its guardrail, `agent.py` is the optional tool-using agent, `llm.py` is the only place a model is called, `evaluate.py` is the evaluation, and `main.py` is the command line. `server.py` and `static/` are the website.

## Setup

You need Python 3.10 or newer.

```bash
# 1. Get the code and enter the folder
cd VirtueWeave

# 2. (Recommended) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install the requirements
pip install -r requirements.txt

# 4. (Optional) turn on the written explanations from the model
export ANTHROPIC_API_KEY="your-key"
```

The app works **without** an API key. It then shows the catalog's own "why it fits" text instead of the model's wording.

**Run the website**

```bash
python3 -m uvicorn server:app --port 8000
```

Open http://localhost:8000, choose a topic, grade and minutes, and press **Recommend**.

**Or use the command line**

```bash
python3 -m src.main --topic loops --grade 1 --minutes 15 --note "they keep giving up"
```

**Try agent mode** (needs an API key)

```bash
python3 -m src.main --topic loops --grade 1 --minutes 15 --note "they keep giving up" --agent
python3 -m src.agent_demo               # runs 3 examples and saves the traces to assets/agent_traces/ and ai_interactions.md
python3 -m src.compare_specialization   # measures baseline vs few-shot wording and writes the table into model_card.md
```

**Run the tests and the evaluation**

```bash
python3 -m pytest                     # 112 tests
python3 -m src.evaluate               # 12-case evaluation, offline (no model calls)
python3 -m src.evaluate --live        # same cases with the real model (costs money)
```

## Sample interactions

These three runs were produced with the command line **without an API key**, so the "How it fits" line is the catalog's own text. With a key, that line becomes two or three sentences written by the model.

**1. Loops, grade 1, 15 minutes, note: "they keep giving up"**

```
#1  Perseverance  (score 1.90)
    How it fits: A loop repeats the same small effort until the goal is reached.
    Try "The Cat That Kept Trying" (15 min): Children use a Repeat block to move a character a little each time until it arrives, and tell the story of not giving up.

#2  Teamwork  (score 1.56)
    How it fits: A repeating pattern made by two people only works if they keep in step.
    Try "Our Dance Loop" (10 min): Pairs invent a three-move dance, perform it as a loop, then program it with a Repeat block.

#3  Gratitude  (score 0.60)
    How it fits: Some good actions are worth repeating every day.
    Try "Thank-You on Repeat" (5 min): Children name a thank-you they say every day and program a character to say it in a loop.
```

**2. Debugging, grade 9, 30 minutes, note: "some blame their partner when it breaks"**

```
#1  Honesty  (score 1.30)
    How it fits: Fixing a bug starts with saying plainly what went wrong.
    Try "I Found My Bug" (30 min): Students write a candid post-mortem of a bug they caused, covering cause, impact and fix.

#2  Teamwork  (score 1.10)
    How it fits: Two people often find a bug faster than one, because each notices what the other misses.
    Try "Two Heads Find Bugs" (30 min): Teams run paired debugging sessions with set roles and reflect on what each role caught.

#3  Perseverance  (score 1.00)
    How it fits: Finding the step that fails and fixing it means trying again with a plan.
    Try "Bug Hunt Try Again" (30 min): Students tackle a hard bug with a documented hypothesis-and-test cycle and reflect on staying with the problem.
```

The note about blaming moved Honesty to the top. Without a note, Perseverance would lead for debugging.

**3. Patterns, kindergarten, 5 minutes, no note**

```
#1  Honesty  (score 0.60)
    How it fits: A pattern is only useful if we describe what we really see.
    Try "Say What You See" (5 min): Children describe an object's features exactly before deciding where it goes.

#2  Curiosity  (score 1.00)
    How it fits: Finding a pattern begins with noticing what is the same and what is different.
    Try "Pattern Detectives" (10 min)  [runs 10 min, over your 5]: Children guess a hidden sorting rule and test the guess with new objects.

#3  Justice  (score 0.80)
    How it fits: A sorting rule can leave something out, and a fair rule has a place for everything.
    Try "Does the Rule Fit Everyone?" (15 min)  [runs 15 min, over your 5]: Children sort objects by a rule, find one that does not fit and change the rule.
```

Only one activity fits in five minutes, so it comes first. The other two are shown with a clear warning that they run longer than the time available.

## Design decisions and trade-offs

- **A fixed topic x virtue table instead of free generation.** Every recommendation traces back to a row someone can read and correct. That makes the system predictable and checkable. The trade-off is scope: it covers 12 topics and 8 virtues, and the content needs expert review. It cannot handle a topic that is not in the table.
- **The model only rewords facts it is given.** It receives the catalog text, the activity and the scorer's reasons, and its output goes through a guardrail with a fallback. This keeps the model from inventing activities, and it means the whole app works without a model. The trade-off is that the wording is less personal than free generation would be.
- **TF-IDF retrieval instead of embeddings.** It is deterministic, needs no extra service, and every match can be explained. The trade-off is that it only matches words. A small stemmer, a cue-tag bonus, a similarity floor and a list of stop words such as "class" and "kids" patch that, but a note phrased in unusual words can still be missed.
- **Four grade bands instead of thirteen grades.** 384 activities was a size I could write and review. The trade-off is that grades 3 and 5 get the same activity.
- **Time handling.** Activities that fit the time come first, and longer ones are flagged. The trade-off is that a lower-fit virtue can outrank a better one when time is short, as in sample 3.
- **No `temperature=0`.** I wanted deterministic model output, but the current API rejects sampling parameters on every model. I used a JSON schema, a short fixed prompt and low effort instead. The model's wording can still vary between runs, which is why the guardrail checks the shape and not the exact text.
- **Plain HTML and JavaScript with a small FastAPI server, not a notebook-style app.** It gave me full control of the look and made the page easy to test. It cost more code.
- **A trace for every run.** Each run writes a JSON file with every step's input, output and timing, which made debugging and evaluation much easier.

## Testing summary

**What I built.** 112 automated tests cover the catalog checks, retrieval, the pipeline (including every guardrail and fallback path), the web API and the model wrapper. A separate 12-case evaluation (`src/evaluate.py`) runs the whole pipeline: 5 ordinary requests with the virtue that should come first, 2 vague notes that must not change the default order, and 5 guardrail cases (4 inputs that must be blocked, 1 where the model's wording is bad and the fallback must appear).

**Results (offline):** 12 of 12 cases pass. The expected virtue is first in 5 of 5 ordinary cases, there are 0 constraint violations (always 3 distinct virtues, the right grade band, correct time flags), and repeat runs give identical results.

**What worked.** The guardrails and fallbacks held in every test. The evaluation caught real problems that unit tests alone would not have.

**What didn't work at first, and what I learned.**
- The first evaluation passed 10 of 12. The teasing and blaming cases ranked the wrong virtue first because the note's words ("tease", "blame") did not match the data's words ("teasing", "blaming"). A small stemmer and a cue-tag bonus fixed it, but I tuned that on the same five cases, so 5 of 5 may be optimistic.
- The vague-note cases failed because a one-word note like "class" matched some activities by accident. A similarity floor and stop words fixed it.
- One of my own test inputs was wrong ("thanks" really does match gratitude), which reminded me to question the test as well as the code.
- A bug that made the cards render as `[object HTMLElement]` only showed up when I looked at a screenshot of the page.

**What I have not tested.** The model's own wording has not been run against a real model, because the evaluation and sample outputs here were produced without an API key. "Repeat runs give the same result" is trivially true offline, so it only becomes a real test with `--live`. The 60 newest topic x virtue entries and 348 activities are AI-drafted and have not been reviewed line by line.

## Reflection

This project taught me to build the deterministic part first and to give the model a narrow job. Once the data, the ranking and the guardrails worked on their own, the model became an optional improvement instead of a risk, and I could see exactly what each part contributed. It also taught me that evaluation changes what you build: the cases I wrote after the first version exposed the matching and noise problems, and I would not have found them by reading the code. Finally, working with a table of AI-drafted content made clear how much of "AI quality" is really data quality, and that someone has to review it.

My responsible-AI reflection, covering how I worked with AI, a suggestion I accepted and one I rejected or revised, and the system's limitations, is in [model_card.md](model_card.md).

## Project layout

```
VirtueWeave/
├── data/            virtues.csv, virtue_fit.csv, activities.csv
├── src/             catalog, retrieval, pipeline, agent, explainer, llm, evaluate, main
├── static/          the website (index.html, style.css, app.js)
├── tests/           112 automated tests
├── assets/          flowchart.mmd (the diagram above)
├── server.py        the web server
└── traces/          one JSON trace per run, plus feedback.jsonl (git-ignored)
```
