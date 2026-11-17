"use strict";

// icon and colour for each virtue; the one-line meaning comes from the server
const VIRTUES = {
  service: ["🤲", "#3b82f6"], compassion: ["💛", "#e0558c"], gratitude: ["🙏", "#d98f00"],
  justice: ["⚖️", "#8b5cf6"], perseverance: ["🧗", "#e8602c"], curiosity: ["🔍", "#0b8fb0"],
  teamwork: ["🧩", "#1f9a4f"], honesty: ["🗝️", "#5b6ee1"],
};
const EXAMPLE = { topic: "loops", grade: "1", minutes: 15, note: "they keep giving up" };
const PROGRESS = ["Checking your answers", "Weighing the eight virtues", "Choosing the best three", "Writing how to bring them in"];

const $ = (selector) => document.querySelector(selector);
const form = $("#ask");
const sendBtn = $("#send");
const resultsSection = $("#results");
const resultsBody = $("#results-body");
let busy = false;
let meanings = {};

const cap = (text) => text[0].toUpperCase() + text.slice(1);
const gradeName = (g) => (g === "K" ? "Kindergarten" : `Grade ${g}`);

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

/* ---------- the form ---------- */
async function fillForm() {
  try {
    const res = await fetch("/api/catalog");
    if (!res.ok) throw new Error("catalog " + res.status);
    const catalog = await res.json();
    meanings = catalog.virtues;
    $("#topic").append(...catalog.topics.map((t) => h("option", { value: t, text: cap(t) })));
    $("#grade").append(...catalog.grades.map((g) => h("option", { value: g, text: gradeName(g) })));
    $("#topic").value = "loops";
    $("#grade").value = "3";
  } catch {
    showResults(h("div", { class: "message" }, h("p", { text: "Couldn't load the topics. Please refresh the page." })));
  }
}

$("#example").addEventListener("click", () => {
  $("#topic").value = EXAMPLE.topic; $("#grade").value = EXAMPLE.grade;
  $("#minutes").value = EXAMPLE.minutes; $("#note").value = EXAMPLE.note;
});

/* ---------- results ---------- */
function showResults(...nodes) {
  resultsSection.hidden = false;
  resultsBody.replaceChildren(...nodes.flat(Infinity).filter(Boolean));
}

function jumpToResults() {
  const smooth = !matchMedia("(prefers-reduced-motion: reduce)").matches;
  resultsSection.scrollIntoView({ behavior: smooth ? "smooth" : "auto", block: "start" });
}

function showProgress() {
  $("#results-title").textContent = "Finding your virtues…";
  const items = PROGRESS.map((label) => h("li", {}, h("span", { class: "dot" }), h("span", { text: label })));
  showResults(h("ul", { class: "steps", "aria-label": "Working on it" }, items));
  jumpToResults();
  let i = 0;
  const tick = () => {
    items.forEach((li, n) => li.className = n < i ? "done" : n === i ? "active" : "");
    if (i < items.length - 1) i += 1;
  };
  tick();
  const timer = setInterval(tick, 1100);
  return () => clearInterval(timer);
}

function renderCard(v, data) {
  const [icon, color] = VIRTUES[v.virtue] || ["✨", "#008037"];
  const status = h("span", { class: "thanks", "aria-live": "polite" });
  const buttons = [];
  const choose = async (decision, label) => {
    buttons.forEach((b) => b.disabled = true);
    try {
      const res = await fetch("/api/feedback", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trace_id: data.trace_id, virtue: v.virtue, decision }),
      });
      status.textContent = res.ok ? label : "Couldn't save that, sorry.";
    } catch { status.textContent = "Couldn't save that, sorry."; }
  };
  buttons.push(
    h("button", { class: "btn small", type: "button", text: "I'll use this", onclick: () => choose("accept", "Great, noted!") }),
    h("button", { class: "btn small ghost", type: "button", text: "Not for me", onclick: () => choose("reject", "Thanks, noted.") }),
  );

  const card = h("article", { class: "card", "aria-label": v.virtue },
    h("div", { class: "card-head" },
      h("div", { class: "badge", "aria-hidden": "true", text: icon }),
      h("div", {}, h("h3", { text: cap(v.virtue) }))),
    h("div", { class: "block" },
      h("p", { class: "label", text: "Why it fits" }),
      h("p", { text: v.fits })),
    h("div", { class: "block" },
      h("p", { class: "label", text: "How to bring it in" }),
      h("div", { class: "how" },
        h("h4", { text: v.title }),
        h("p", { class: "time", text: `${v.minutes} min · ${gradeName(data.request.grade)}` }),
        h("p", { text: v.activity }),
        v.over_time ? h("p", { class: "over",
          text: `This runs ${v.minutes} minutes, longer than the ${data.request.minutes} you have. Try a shorter version.` }) : null)),
    h("div", { class: "actions" }, buttons, status));
  card.style.setProperty("--v", color);
  return card;
}

function renderResult(data) {
  if (!data.ok) {
    $("#results-title").textContent = "Let's try that again";
    showResults(h("div", { class: "message" }, h("p", { text: data.message || "I couldn't find a match. Check the topic, grade and minutes." })));
    return;
  }
  const r = data.request;
  $("#results-title").textContent = `Three virtues to weave into ${cap(r.topic)}`;
  const bits = [gradeName(r.grade), `${r.minutes} minutes`];
  if (r.note) bits.push(`“${r.note}”`);
  showResults(
    h("p", { class: "subtitle", text: bits.join(" · ") }),
    h("div", { class: "cards" }, data.virtues.map((v) => renderCard(v, data))));
}

/* ---------- sending ---------- */
async function send(payload) {
  busy = true;
  sendBtn.disabled = true;
  const stop = showProgress();
  try {
    const res = await fetch("/api/recommend", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("server " + res.status);
    const data = await res.json();
    stop();
    renderResult(data);
  } catch {
    stop();
    $("#results-title").textContent = "Something went wrong";
    showResults(h("div", { class: "message" },
      h("p", { text: "Sorry, something went wrong on our side." }),
      h("div", { class: "actions" },
        h("button", { class: "btn small", type: "button", text: "Try again", onclick: () => !busy && send(payload) }))));
  } finally {
    busy = false;
    sendBtn.disabled = false;
    $("#results-title").focus({ preventScroll: true });
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (busy) return;
  send({
    topic: $("#topic").value, grade: $("#grade").value,
    minutes: parseInt($("#minutes").value, 10) || 0, note: $("#note").value.trim(),
  });
});

/* ---------- theme ---------- */
const themeBtn = $("#theme");
const themeIcon = $("#theme-icon");
const isDark = () => {
  const forced = document.documentElement.dataset.theme;
  return forced ? forced === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
};
const paintThemeIcon = () => { themeIcon.textContent = isDark() ? "☀️" : "🌙"; };
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

fillForm();
