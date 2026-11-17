"""Writes a generated section into a markdown file between two marker comments."""

from pathlib import Path


def replace_section(path: str, name: str, markdown: str) -> None:
    """
    Puts markdown between <!-- name:start --> and <!-- name:end --> in the file,
    replacing what was there. If the markers are missing, the section is added
    at the end. Everything outside the markers is left alone, so running a
    script twice gives the same file.
    """
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    block = f"{start}\n{markdown.strip()}\n{end}"
    file = Path(path)
    text = file.read_text(encoding="utf-8") if file.exists() else ""
    if start in text and end in text:
        head, rest = text.split(start, 1)
        _, tail = rest.split(end, 1)
        text = head + block + tail
    else:
        text = text.rstrip("\n") + ("\n\n" if text.strip() else "") + block + "\n"
    file.write_text(text, encoding="utf-8")
