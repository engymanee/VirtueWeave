# VirtueWeave: model card

**What it is.** A virtue recommender for computer science teachers. Given a topic, a grade (K-12), the minutes available and an optional note about the class, it recommends the three virtues that fit best and shows how to bring each one into the lesson. A fixed table of 96 topic x virtue pairs and 384 grade-level activities does the matching. A language model only rewords the table's own text, and a guardrail checks its wording. See the [README](README.md) for the architecture and test results.

**Intended use.** Giving a teacher ideas while planning a lesson. It is not meant to judge, label or grade students.

## Limitations and biases

- **The content is one point of view.** The fit scores and activities reflect one author's judgment and AI drafts. The 60 newest pairs and 348 activities have not been reviewed line by line.
- **Ties favour list order.** Many topics tie at a fit of 0.6, and a tie goes to whichever virtue is listed first (service, compassion, ...), not to the better one.
- **Eight fixed virtues is a particular framing.** The virtue list, the English-only text and the K-12 grade bands are not culturally neutral.
- **Matching is by words only.** A note phrased in unusual words can be missed. I tuned the matching on five cases, and the evaluation has only 12, so its scores may be optimistic.
- **Time can beat fit.** When time is short, a lower-fit virtue can outrank a better one.
- **Model wording is untested.** I have not run it against a real model.

## Misuse and how I would reduce it

- **Labelling children.** A teacher could use the output to judge individual students. The page says it suggests ideas and the teacher decides, and it never takes student data.
- **Personal details in the note.** Names typed into the note go to the model API if a key is set, and they are saved in local traces. I have not yet added a "don't include student names" warning next to the box; that is the next change.
- **Prompt injection.** A note could try to instruct the assistant. A guardrail blocks the common phrasings, but it is a simple pattern check, not a strong defence.
- **Cost abuse.** If the site were public with a key set, anyone could trigger model calls. There is no rate limit yet.

## What surprised me while testing

- The first evaluation scored 10 of 12, which showed how much word matching and noise matter.
- A one-word note such as "class" reordered the results until I added a similarity floor and stop words.
- The model API rejected `temperature`, so I could not make the model deterministic that way.
- One of my own test inputs was wrong ("thanks" really does match gratitude).
- The app felt complete without the model, which showed how much of the quality comes from the data.

## Future improvements

- **Generate a lesson plan.** After a teacher picks a virtue, the model would turn the chosen activity into a short plan for that grade and time: an opening question, the steps with timings, discussion questions that name the virtue, and a closing reflection. It would be built only from the catalog's activity and "why it fits" text, and it would go through the same kind of output check as the current explanations, so it cannot invent activities.
- **Let teachers suggest improvements.** Teachers could add their own ideas: a better activity for a topic and grade, a correction to a "why it fits" sentence, or a new virtue. The accept and reject buttons already save to `traces/feedback.jsonl`; a free-text box would sit beside them. Suggestions would go into a review queue and enter the catalog only after a person approves them, so the model never changes itself from unchecked input. Over time the accept and reject counts could also be used to review the fit scores.
- **Review the content with experts.** Have teachers and character-education experts check the AI-drafted pairs and activities, and record the results.
- **Measure the model live.** Run the evaluation and the baseline-versus-few-shot comparison with a real model.
- **Safer public use.** Add a "don't include student names" warning next to the note box and a rate limit on the server.

## How I worked with AI

I built this with Claude Code in VS Code. I set the direction and made the product decisions: turning my music recommender into a virtue recommender, choosing a form (topic, grade, minutes, optional note) over free text, and pushing the interface to be simpler. Claude wrote the code, tests and draft content, and I checked the results.

- **A suggestion I accepted:** a full topic x virtue table. My data had exactly three virtues per topic, so "recommend three virtues" could only ever return those three. Claude recommended a table of all 96 pairs, and I accepted it because it fixed that.
- **A suggestion I changed:** Claude added a "Why we picked this" and a "Where this comes from" panel to every answer, showing the scoring reasons and source passages. That was more detail than a teacher needs and it made the answers harder to read, so I had them removed and the page kept only why it fits and how to bring it in. Claude also recommended limiting the first version to grades K-2, and I widened it to K-12 so any teacher could use it.

<!-- specialization:start -->
## Specialization: baseline vs few-shot examples

The explainer's prompt includes two few-shot examples that set a warm, plain tone, and they can be switched off. **The measured comparison has not been run yet**, because it needs a real model. Run `python3 -m src.compare_specialization` with an API key and this section is replaced with the results (guardrail pass rate, sentences per virtue, words per sentence, how often the activity is named, and how often the teacher's note is used).
<!-- specialization:end -->
