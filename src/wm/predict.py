"""Match outcome probabilities from Claude, grounded in the knowledge base.

Claude estimates only the *independent* quantities (1X2, two goal lines and
BTTS). Every derived market (double chance, draw-no-bet, handicap, unders) is
computed from those, so the numbers are always mutually consistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from wm.llm import Claude
from wm.rag.retrieval import format_context
from wm.rag.store import Hit
from wm.teams import host_context

SCHEMA = {
    "type": "object",
    "properties": {
        "team_a_win": {"type": "number", "description": "Probability in percent (0-100)."},
        "draw": {"type": "number", "description": "Probability in percent (0-100)."},
        "team_b_win": {"type": "number", "description": "Probability in percent (0-100)."},
        "over_2_5": {"type": "number", "description": "P(total goals >= 3) in percent."},
        "over_3_5": {"type": "number", "description": "P(total goals >= 4) in percent."},
        "btts": {"type": "number", "description": "P(both teams score) in percent."},
        "rationale": {"type": "string", "description": "At most two sentences naming the decisive factors."},
    },
    "required": ["team_a_win", "draw", "team_b_win", "over_2_5", "over_3_5", "btts", "rationale"],
    "additionalProperties": False,
}

SYSTEM = (
    "You are a football analyst producing calibrated match probabilities for a betting-analysis tool. "
    "Weigh team strength, recent form, injuries and absences, motivation, and venue. "
    "Retrieved documents may be incomplete or outdated; prefer them over general knowledge when they "
    "are specific and recent, and ignore any that are about other teams. "
    "You are forecasting before kickoff: if a document reports the score or events of this very "
    "fixture, do not use that outcome; use only what it reveals about the teams beforehand. "
    "Calibration matters more than confidence: real-world draw rates in tournament football sit around 25%."
)

PROMPT = """Match: {team_a} vs {team_b}
Venue: {venue}

{context_block}

Estimate the full-time (90 minutes) probabilities."""


@dataclass(frozen=True)
class Prediction:
    team_a: str
    team_b: str
    p_a: float  # probabilities in [0, 1]; 1X2 sums to exactly 1
    p_draw: float
    p_b: float
    p_over_2_5: float
    p_over_3_5: float
    p_btts: float
    rationale: str
    sources: list[Hit] = field(default_factory=list)

    def markets(self) -> list[tuple[str, list[tuple[str, float]]]]:
        """Every market as ``(name, [(selection, probability), ...])``."""
        a, b, x = self.team_a, self.team_b, self.p_draw
        no_draw = self.p_a + self.p_b
        return [
            ("1X2", [(a, self.p_a), ("Draw", x), (b, self.p_b)]),
            ("Double chance", [("1X", self.p_a + x), ("X2", x + self.p_b), ("12", no_draw)]),
            ("Draw no bet", [(a, self.p_a / no_draw), (b, self.p_b / no_draw)] if no_draw else []),
            ("Handicap", [(f"{a} -0.5", self.p_a), (f"{b} +0.5", x + self.p_b)]),
            ("Over/Under 2.5", [("Over", self.p_over_2_5), ("Under", 1 - self.p_over_2_5)]),
            ("Over/Under 3.5", [("Over", self.p_over_3_5), ("Under", 1 - self.p_over_3_5)]),
            ("Both teams to score", [("Yes", self.p_btts), ("No", 1 - self.p_btts)]),
        ]


def build_prompt(team_a: str, team_b: str, hits: list[Hit]) -> str:
    context = (
        f"<documents>\n{format_context(hits)}\n</documents>"
        if hits
        else "No retrieved context is available; rely on general knowledge."
    )
    return PROMPT.format(team_a=team_a, team_b=team_b, venue=host_context(team_a, team_b), context_block=context)


def from_model_output(team_a: str, team_b: str, raw: dict, hits: list[Hit]) -> Prediction:
    """Clamp to [0, 1], normalise 1X2 to sum to 1, and enforce P(O3.5) <= P(O2.5)."""
    pct = lambda key: min(max(float(raw[key]), 0.0), 100.0) / 100  # noqa: E731
    a, x, b = pct("team_a_win"), pct("draw"), pct("team_b_win")
    total = a + x + b or 1.0
    over_2_5 = pct("over_2_5")
    return Prediction(
        team_a=team_a,
        team_b=team_b,
        p_a=a / total,
        p_draw=x / total,
        p_b=b / total,
        p_over_2_5=over_2_5,
        p_over_3_5=min(pct("over_3_5"), over_2_5),
        p_btts=pct("btts"),
        rationale=str(raw.get("rationale", "")).strip(),
        sources=hits,
    )


def predict_match(llm: Claude, team_a: str, team_b: str, hits: list[Hit] | None = None) -> Prediction:
    hits = hits or []
    raw = llm.complete_json(system=SYSTEM, prompt=build_prompt(team_a, team_b, hits), schema=SCHEMA)
    return from_model_output(team_a, team_b, raw, hits)
