"""The World Cup 2026 fixture list with real 90-minute results.

``fetch_fixtures`` pulls it from API-Football once and ``save_fixtures`` writes
it to ``eval/wc2026_results.csv``, which is committed so every later run (and
CI) scores against the same ground truth without network access.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

from wm.teams import TEAM_NAME_ALIASES, all_teams, normalize

WORLD_CUP_LEAGUE_ID = 1
SEASON = 2026

# API-Football status codes for a finished match (regular time, extra time, penalties).
FINISHED = {"FT", "AET", "PEN"}
WENT_PAST_90 = {"AET", "PEN"}


@dataclass(frozen=True)
class Fixture:
    match_id: str
    kickoff: str  # ISO 8601, UTC
    stage: str  # "Group Stage - 1", "Round of 32", ..., "Final"
    team_a: str  # API home side; the tournament is (almost entirely) on neutral ground
    team_b: str
    goals_a: int  # after 90 minutes plus stoppage time: extra time and penalties are excluded
    goals_b: int
    status: str

    @property
    def outcome(self) -> int:
        """0 = team_a win, 1 = draw, 2 = team_b win (after 90 minutes)."""
        return 0 if self.goals_a > self.goals_b else 1 if self.goals_a == self.goals_b else 2

    @property
    def total_goals(self) -> int:
        return self.goals_a + self.goals_b

    @property
    def btts(self) -> bool:
        return self.goals_a > 0 and self.goals_b > 0

    @property
    def is_group(self) -> bool:
        return self.stage.lower().startswith("group")

    @property
    def kickoff_at(self) -> datetime:
        return datetime.fromisoformat(self.kickoff)


def _official_names() -> dict[str, str]:
    """normalised API-Football name or alias -> the tournament name used everywhere else in wm."""
    names = {normalize(t): t for t in all_teams()}
    for official, aliases in TEAM_NAME_ALIASES.items():
        names.update({normalize(a): official for a in aliases})
    return names


def official_name(api_name: str) -> str:
    return _official_names().get(normalize(api_name), api_name)


def parse_api_fixtures(items: list[dict]) -> tuple[list[Fixture], list[str]]:
    """Convert API-Football ``/fixtures`` items. Returns (finished fixtures, warnings)."""
    fixtures, warnings = [], []
    known = set(_official_names().values())
    for item in items:
        fx, teams, score = item.get("fixture", {}), item.get("teams", {}), item.get("score", {})
        status = fx.get("status", {}).get("short", "")
        label = f"{teams.get('home', {}).get('name')} vs {teams.get('away', {}).get('name')}"
        if status not in FINISHED:
            warnings.append(f"skipped {label}: status {status or 'unknown'}")
            continue
        fulltime = score.get("fulltime") or {}
        goals_a, goals_b = fulltime.get("home"), fulltime.get("away")
        if goals_a is None or goals_b is None:
            warnings.append(f"skipped {label}: no full-time score")
            continue
        # A match only goes to extra time if it was level after 90 minutes. If not, the API's
        # "fulltime" field includes extra time and the 1X2 outcome would be wrong.
        if status in WENT_PAST_90 and goals_a != goals_b:
            raise ValueError(f"{label} ({status}) has a non-level 90-minute score {goals_a}-{goals_b}")
        team_a, team_b = official_name(teams["home"]["name"]), official_name(teams["away"]["name"])
        for name in (team_a, team_b):
            if name not in known:
                warnings.append(f"unknown team name {name!r}: add it to TEAM_NAME_ALIASES")
        fixtures.append(
            Fixture(
                match_id=str(fx["id"]),
                kickoff=datetime.fromisoformat(fx["date"]).isoformat(),
                stage=str(item.get("league", {}).get("round", "")),
                team_a=team_a,
                team_b=team_b,
                goals_a=int(goals_a),
                goals_b=int(goals_b),
                status=status,
            )
        )
    fixtures.sort(key=lambda f: (f.kickoff, f.match_id))
    return fixtures, warnings


def fetch_fixtures(client) -> tuple[list[Fixture], list[str]]:
    """Every finished World Cup 2026 match from API-Football (one request)."""
    return parse_api_fixtures(client.get("/fixtures", league=WORLD_CUP_LEAGUE_ID, season=SEASON))


_COLUMNS = [f.name for f in fields(Fixture)]


def save_fixtures(fixtures: list[Fixture], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(asdict(f) for f in fixtures)


def load_fixtures(path: str | Path) -> list[Fixture]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run `wm eval fetch` first.")
    with path.open(newline="", encoding="utf-8") as fh:
        return [
            Fixture(**{**row, "goals_a": int(row["goals_a"]), "goals_b": int(row["goals_b"])})
            for row in csv.DictReader(fh)
        ]
