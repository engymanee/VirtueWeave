"""Scripted fake-model pieces shared by the agent tests."""

from types import SimpleNamespace


def say(text):
    return SimpleNamespace(type="text", text=text)


def use(name, call_id="t", **args):
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=args)


def reply(*blocks):
    return SimpleNamespace(content=list(blocks), stop_reason="tool_use")


class Script:
    """A fake model: returns the prepared replies in order and records what it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, system, messages, tools):
        self.calls.append([dict(m) for m in messages])
        return self.replies.pop(0)


GOOD = [
    reply(say("First I will look at the topic."),
          use("get_topic_fits", "a", reason="see how each virtue fits")),
    reply(use("search_class_notes", "b", query="unfair turns", reason="read the note")),
    reply(use("get_activity", "c", virtue="justice", reason="check the time")),
    reply(use("submit_recommendation", "d", virtues=["justice", "teamwork", "perseverance"],
              reasoning="Justice matches the note; the others fit the topic.")),
]
