"""Use Claude to turn a raw match report into structured, retrieval-friendly notes."""

from __future__ import annotations

import re

from wm.llm import Claude

HEADINGS = [
    "Match",
    "Teams",
    "Result",
    "Winner",
    "Goals And Scorers",
    "Cards",
    "Injuries And Absences",
    "Team Control And Passing",
    "Chance Creation And Finishing",
    "Defensive Notes",
    "Tactical Or Momentum Notes",
    "Player Notes",
    "Prediction-Relevant Takeaways",
    "Uncertain Or Missing Details",
]

SYSTEM = (
    "You are a football match analyst preparing structured context for a retrieval database "
    "that will later inform match predictions. Extract only facts that the report states or "
    "strongly implies. If a detail is missing, write 'not stated' rather than guessing: "
    "invented statistics, injuries, scorers or cards would corrupt later predictions."
)

PROMPT = """Analyze this recent football match report.

<report>
{text}
</report>

Start with one line in exactly this form (it labels every stored chunk, so leave the score out):
Title: <home team> vs <away team>, <competition>, <date>

Then write plain-text notes under exactly these headings, in this order:
{headings}"""

_MARKUP = "#*_`- "


def _clean(line: str) -> str:
    return re.sub(r"[*_`]", "", line).strip(_MARKUP)


def match_title(notes: str) -> str | None:
    """A short label for a match analysis, e.g. "Canada vs Bosnia and Herzegovina".

    Uses the ``Title:`` line that current analyses start with. For analyses
    written before that existed, it builds "Home vs Away" from the ``Teams``
    section instead.
    """
    raw_lines = notes.splitlines()
    lines = [_clean(line) for line in raw_lines]
    for line in lines:
        if line.lower().startswith("title:") and line[6:].strip():
            return line[6:].strip()

    headings = {h.lower() for h in HEADINGS}
    for i, (raw, line) in enumerate(zip(raw_lines, lines, strict=True)):
        # Only a real heading ("## Teams" or a bare "Teams" line), not "Teams" inside prose.
        is_heading = raw.lstrip().startswith("#") or raw.strip().rstrip(":") == "Teams"
        if not is_heading or line.lower().rstrip(":") != "teams":
            continue
        names: list[str] = []
        for nxt in lines[i + 1 :]:
            if not nxt or set(nxt) == {"-"}:
                continue
            if nxt.lower().rstrip(":") in headings:
                break
            if " vs " in nxt:  # "Mexico vs South Africa"
                return nxt
            label, sep, value = nxt.partition(":")
            names.append(value.strip() if sep and len(label) <= 12 else nxt)  # "Home: Canada", "Team A: Korea"
            if len(names) == 2:
                return f"{names[0]} vs {names[1]}"
        return None
    return None


def analyze_report(llm: Claude, text: str) -> str:
    """Structured notes followed by the original report, ready to be stored."""
    notes = llm.complete(
        system=SYSTEM,
        prompt=PROMPT.format(text=text, headings="\n".join(HEADINGS)),
    )
    return f"{notes}\n\nOriginal Match Report\n{text.strip()}"
