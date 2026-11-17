"use strict";

const EXAMPLES = [
  "Teaching loops to my 6-year-olds tomorrow, 15 minutes, they keep giving up.",
  "My 7-year-olds argue over who gets the tablet during pair programming, 10 minutes.",
  "Sharing projects with parents on Friday, 20 minutes, ages 6, the kids never say thank you.",
  "Teaching conditionals to 6-year-olds, 10 minutes, a few kids tease classmates who get it wrong.",
];

// Colour and icon for each virtue (colour is mixed with the page colours in CSS).
const VIRTUES = {
  service: ["🤲", "#3b82f6"], compassion: ["💛", "#ec5f95"], gratitude: ["🙏", "#f0a312"],
  justice: ["⚖️", "#8b5cf6"], perseverance: ["🧗", "#ef6b3b"], curiosity: ["🔍", "#0ea5c6"],
  teamwork: ["🧩", "#2fa55b"], honesty: ["🗝️", "#5b6ee1"],
};

const STEP_LABELS = {
  guard_input: "Checked your message", interpret: "Understood your class",
  retrieve: "Searched the practice notes", rank: "Picked the best fits",
  explain: "Wrote why they fit", guard_output: "Checked the wording",
};
const PROGRESS = [
  "Reading your message", "Searching the practice notes",
  "Picking the best fits", "Writing why they fit",
];

const log = document.querySelector("#log");
const form = document.querySelector("#composer");
const input = document.querySelector("#sentence");
const sendBtn = document.querySelector("#send");
let busy = false;

/* Builds elements with textContent only, so nothing from the server is parsed as HTML. */
function h(tag, props = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value == null) continue;
    if (key === "class") el.className = value;
    else if (key === "text") el.textContent = value;
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else el.setAttribute(key, value);
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid != null) el.append(kid);
  }
  return el;
}

function avatar() {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "avatar");
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", "#weave");
  svg.append(use);
  return svg;
}

function scrollDown() {
  const smooth = !matchMedia("(prefers-reduced-motion: reduce)").matches;
  window.scrollTo({ top: document.body.scrollHeight, behavior: smooth ? "smooth" : "auto" });
}

function addUser(text) {
  const row = h("div", { class: "row user" }, h("div", { class: "bubble" }, h("p", { text })));
  log.append(row);
  scrollDown();
}

function addAssistant(content, wide = false) {
  const row = h("div", { class: "row assistant" }, avatar(),
    h("div", { class: "bubble" + (wide ? " wide" : "") }, content));
  log.append(row);
  scrollDown();
  return row;
}

/* ---------- greeting ---------- */
function greet() {
  addAssistant([
    h("p", { text: "Hi, I'm VirtueWeave! 👋" }),
    h("p", { text: "Tell me about your class in a sentence or two: the coding topic, how old the " +
      "children are, how long you have, and what has been happening. I'll suggest three " +
      "practices that teach coding and help children grow a virtue too." }),
    h("div", { class: "chips", "aria-label": "Example sentences" },
      EXAMPLES.map((text) => h("button", {
        class: "chip", type: "button", text,
        onclick: () => { input.value = text; resize(); input.focus(); },
      }))),
  ]);
}

/* ---------- progress while the server works ---------- */
function showProgress() {
  const items = PROGRESS.map((label) =>
    h("li", {}, h("span", { class: "dot" }), h("span", { text: label })));
  const list = h("ul", { class: "steps", "aria-label": "Working on it" }, items);
  const row = addAssistant(list);
  let i = 0;
  const tick = () => {
    items.forEach((li, n) => li.className = n < i ? "done" : n === i ? "active" : "");
    if (i < items.length - 1) i += 1;
  };
  tick();
  const timer = setInterval(tick, 1100);
  return () => { clearInterval(timer); row.remove(); };
}

/* ---------- results ---------- */
function meta(text) { return h("span", { class: "chip small", text }); }

function renderPractice(p, traceId) {
  const status = h("span", { class: "thanks", "aria-live": "polite" });
  const buttons = [];
  const choose = async (decision, label) => {
    buttons.forEach((b) => b.disabled = true);
    try {
      const res = await fetch("/api/feedback", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trace_id: traceId, practice_id: p.id, decision }),
      });
      status.textContent = res.ok ? label : "Couldn't save that, sorry.";
    } catch { status.textContent = "Couldn't save that, sorry."; }
  };
  buttons.push(
    h("button", { class: "btn", type: "button", text: "👍 I'll use this",
      onclick: () => choose("accept", "Great, noted!") }),
    h("button", { class: "btn ghost", type: "button", text: "Not for me",
      onclick: () => choose("reject", "Thanks, noted.") }),
  );

  return h("article", { class: "practice" },
    h("h4", { text: p.title }),
    h("div", { class: "chips" },
      meta(`⏱ ${p.minutes} min`),
      meta(`Ages ${p.age_min}–${p.age_max}`),
      meta(p.topic)),
    h("details", {}, h("summary", { text: "Why this fits" }),
      h("ul", {}, p.reasons.map((r) => h("li", { text: r })))),
    h("details", {}, h("summary", { text: "Where this comes from" }),
      p.sources.map((s) => h("p", { class: "source" },
        h("strong", { text: s.label + ": " }), s.text))),
    h("div", { class: "actions" }, buttons, status));
}

function renderVirtue(v, data) {
  const [icon, color] = VIRTUES[v.virtue] || ["✨", "#0f9d8a"];
  const card = h("section", { class: "virtue", "aria-label": v.virtue },
    h("div", { class: "virtue-head" },
      h("div", { class: "virtue-icon", "aria-hidden": "true", text: icon }),
      h("div", {},
        h("h3", { text: v.virtue[0].toUpperCase() + v.virtue.slice(1) }),
        v.explanation ? h("p", { text: v.explanation }) : null)),
    h("div", { class: "virtue-body" }, v.practices.map((p) => renderPractice(p, data.trace_id))));
  card.style.setProperty("--v", color);
  return card;
}

function renderResult(data) {
  if (!data.ok) {
    addAssistant(h("p", { text: "🙂 " + (data.message || "I couldn't find a match. Try adding the age and how long you have.") }));
    return;
  }
  const u = data.understood || {};
  const chips = [
    u.topic ? meta(`Topic: ${u.topic}`) : h("span", { class: "chip small muted", text: "Topic: not sure" }),
    u.age ? meta(`Age ${u.age}`) : h("span", { class: "chip small muted", text: "Age: not given" }),
    u.minutes ? meta(`${u.minutes} min`) : h("span", { class: "chip small muted", text: "Time: not given" }),
  ];
  const trace = h("details", { class: "trace" }, h("summary", { text: "How I got this" }),
    h("ul", { class: "steps" }, (data.steps || []).map((s) =>
      h("li", { class: s.status === "ok" || s.status === "pass" ? "done" : "" },
        h("span", { class: "dot" }),
        h("span", { text: (STEP_LABELS[s.step] || s.step) + (s.status === "failed" || s.status === "skipped" ? ` (${s.status})` : "") }),
        s.ms != null ? h("span", { class: "ms", text: `${Math.round(s.ms)} ms` }) : null))));

  addAssistant([
    h("p", { text: "Here's what I understood:" }),
    h("div", { class: "understood" }, chips),
    data.explained ? null : h("p", { class: "note",
      text: "I couldn't write a friendly note this time, so I'm showing the reasons behind each pick." }),
    data.virtues.map((v) => renderVirtue(v, data)),
    trace,
  ], true);
}

/* ---------- sending ---------- */
async function send(sentence) {
  busy = true;
  sendBtn.disabled = true;
  addUser(sentence);
  const stop = showProgress();
  try {
    const res = await fetch("/api/ask", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sentence }),
    });
    if (!res.ok) throw new Error("server " + res.status);
    const data = await res.json();
    stop();
    renderResult(data);
  } catch {
    stop();
    addAssistant([
      h("p", { text: "Sorry, something went wrong on my side. 😕" }),
      h("div", { class: "actions" },
        h("button", { class: "btn", type: "button", text: "Try again", onclick: () => !busy && send(sentence) })),
    ]);
  } finally {
    busy = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

function resize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 140) + "px";
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const sentence = input.value.trim();
  if (!sentence || busy) return;
  input.value = "";
  resize();
  send(sentence);
});
input.addEventListener("input", resize);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});

/* ---------- theme ---------- */
const themeBtn = document.querySelector("#theme");
const themeIcon = document.querySelector("#theme-icon");
function isDark() {
  const forced = document.documentElement.dataset.theme;
  return forced ? forced === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
}
function paintThemeIcon() { themeIcon.textContent = isDark() ? "☀️" : "🌙"; }
try {
  const saved = localStorage.getItem("vw-theme");
  if (saved) document.documentElement.dataset.theme = saved;
} catch { /* storage can be blocked; the page still works */ }
themeBtn.addEventListener("click", () => {
  const next = isDark() ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("vw-theme", next); } catch { /* ignore */ }
  paintThemeIcon();
});
paintThemeIcon();

greet();
input.focus();
